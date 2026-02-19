function [R2_test, MSE_test, outICA] = runICAAnalysis(eeg_train, eeg_test, h_train, h_test, num_comps, param, method_dir)
% runICAAnalysis EEGLAB ICA + PCA whitening with detailed Frequency Analysis
%
% Inputs:
%   eeg_train, eeg_test : (Channels x Time)
%   h_train, h_test     : (Time x Latents)
%   num_comps           : Number of ICA components (k)
%   param               : Struct with .fs, .N_F, .f_peak
%   method_dir          : Parent folder for results

%% 1. Setup and Directory
if ~exist(method_dir, 'dir')
    mkdir(method_dir);
end
% File suffix for saving
file_suffix = sprintf('_k%d', num_comps);
h_f_colors = lines(param.N_F); 

%% 2. Run ICA (EEGLAB)
% Prepare EEGLAB Structure
EEG = eeg_emptyset();
EEG.data   = double(eeg_train);
EEG.nbchan = size(eeg_train,1);
EEG.pnts   = size(eeg_train,2);
EEG.trials = 1;
EEG.srate  = param.fs; 
EEG.xmin   = 0;
EEG = eeg_checkset(EEG);

% Rank check
data_rank = rank(EEG.data); 
if num_comps > data_rank
    warning('Requested %d components, but data rank is %d. Capping components.', num_comps, data_rank);
    pca_dim = data_rank;
else
    pca_dim = num_comps;
end

% Run ICA with PCA reduction
try
    % EEG = pop_runica(EEG,'extended', 1, 'pca', pca_dim, 'interrupt','off', 'verbose', 'off'); 
    % FastICA expects (Channels x Time)
    % 'lastEig': performs PCA reduction before ICA
    % 'approach': 'symm' is generally faster and more robust than 'defl'
    % 'g': 'tanh' is a good general purpose non-linearity
    [icasig, A, W] = fastica(eeg_train, ...
        'numOfIC', num_comps, ...
        'lastEig', pca_dim, ...
        'verbose', 'off', ...
        'displayMode', 'off', ...
        'approach', 'symm', ...
        'g', 'tanh');

    % icasig is (Components x Time) -> Transpose to (Time x Comp)
    icasig_train = icasig';

    % Project Test Data
    % FastICA mapping: Source = W * X
    % We apply the same W to the test data
    icasig_test = (W * eeg_test)'; % (Time x Comp)
catch ME
    % warning('Extended ICA failed or not found, trying "runica" default');
    % EEG = pop_runica(EEG, 'pca', pca_dim, 'interrupt','off', 'verbose', 'off'); 
    fprintf('FastICA failed or not installed. Falling back to MATLAB "rica". Error: %s \n', ME.message);
    % Error handling for FastICA failure
    % Fallback: MATLAB Statistics Toolbox "rica"
    Mdl = rica(eeg_train', num_comps); % rica expects (Time x Channels)
    icasig_train = transform(Mdl, eeg_train');
    icasig_test  = transform(Mdl, eeg_test');
end

% % Get Activations (Time x Components)
% icasig_train = double(EEG.icaact)';   
% 
% % Project Test Data
% if ~isempty(EEG.icaweights)
%     % Unmixing matrix * Sphere * Data
%     icasig_test = EEG.icaweights * EEG.icasphere * eeg_test;
%     icasig_test = real(icasig_test)';     % Time x Components
% else
%     icasig_test = zeros(size(eeg_test,2), pca_dim); 
% end

% Mapping components to latents
C = icasig_test;   % dPCA gives nComp x T → transpose to T x nComp
H = h_test(1:size(C,1), :);

[corr_ICA, R_ICA] = match_components_to_latents(C, H, 'ICA', num_comps);


%% 3. Linear Mapping ICs -> Latent Fields
% We want to map the K independent components to the N_F latent fields.
% Using ALL k components available to reconstruct the latent variables.

% % Learn weights W such that: ICs_train * W ≈ H_train
% W_ica = zeros(pca_dim, param.N_F);
% for f = 1:param.N_F
%     W_ica(:,f) = lsqlin(icasig_train, h_train(:,f), [], [], [], [], [], [], [], ...
%                  optimoptions('lsqlin','Display','off'));
% end
% 
% % Reconstruct Test Data: ICs_test * W
% h_rec_test = icasig_test * W_ica;

% OLD WAY (Slow): lsqlin loop
% NEW WAY (Fast): Matrix Left Division (\)
% We solve: icasig_train * W_map = h_train
% W_map = icasig_train \ h_train;

W_map = icasig_train \ h_train;

% Reconstruct Data (Raw)
h_rec_train_raw = icasig_train * W_map;
h_rec_test_raw  = icasig_test * W_map;

% --- NORMALIZATION STEP (MATCHING PCA/dPCA) ---
% Normalize column-wise so std=1 for every feature
h_rec_train = h_rec_train_raw ./ std(h_rec_train_raw, 0, 1);
h_rec_test  = h_rec_test_raw  ./ std(h_rec_test_raw, 0, 1);

% Calculate Metrics
R2_test_global = zeros(1, param.N_F); 
MSE_test_global = zeros(1, param.N_F); 

for f = 1:param.N_F
    % MSE
    MSE_test_global(f) = mean((h_test(:,f) - h_rec_test(:,f)).^2);
    
    % R2
    res_var = sum((h_test(:,f) - h_rec_test(:,f)).^2);
    tot_var = sum((h_test(:,f) - mean(h_test(:,f))).^2);
    R2_test_global(f) = 1 - (res_var / tot_var);
end

% Final Outputs for Main Script (Average across latents for global score)
R2_test = mean(R2_test_global);
MSE_test = mean(MSE_test_global);

% Compute Zero-Lag Correlation for plotting
zeroLagCorr_ica = zeros(1, param.N_F);
for f = 1:param.N_F
    c = corrcoef(h_test(:,f), h_rec_test(:,f));
    zeroLagCorr_ica(f) = c(1,2);
end


%% ============================================================
% PLOTTING SECTION
% ============================================================
if isempty(getCurrentTask()) && num_comps>4
    % Time domain plot
    plotTimeDomainReconstruction(h_test, h_rec_test, param, 'ICA', num_comps, zeroLagCorr_ica, method_dir);
    
    % Independent Component traces plot
    plotCTraces(num_comps, param, h_rec_test, method_dir, file_suffix);

    % Frequency Analysis FFT
    save_path_fft = fullfile(method_dir, ['ICA_FFT_True_vs_Recon' file_suffix '.png']);
    [outFSP] = plotFrequencySpectra(h_test, h_rec_test, 'ICA', param, num_comps, save_path_fft);
    
    nHz = outFSP.nHz;
    Ht = outFSP.Ht;
    Hr = outFSP.Hr;
    Ht_avg_ica = outFSP.Ht_avg;
    Hr_avg_ica = outFSP.Hr_avg;
    R2_avg_ica = outFSP.R2_avg;
    f_freq = outFSP.f_axis;
    f_plot = outFSP.f_plot;
 
    %% Plot 2: Band-wise R2 Bar Chart
   
    br2_path = fullfile(method_dir, ['ICA_Bandwise_R2' file_suffix '.png']);
    [outBR2P] = plotBandwiseR2(R2_avg_ica, f_freq, param, num_comps, 'ICA', br2_path);
    bands = outBR2P.bands;
    band_names = outBR2P.b_names; 
    nBands = outBR2P.nBands;
    %% Plot 3: Scatter plot: True vs Reconstructed Band Amplitudes (Mean)
    Ht_amp_ica = abs(Ht_avg_ica(1:nHz, :));  
    Hr_amp_ica = abs(Hr_avg_ica(1:nHz, :)); 
    
    % Normalizing amplitudes
    Ht_amp_ica = Ht_amp_ica ./ max(Ht_amp_ica(:));
    Hr_amp_ica = Hr_amp_ica ./ max(Hr_amp_ica(:));
    
    mean_band_amp_ica_true  = zeros(nBands, param.N_F);
    mean_band_amp_ica_recon = zeros(nBands, param.N_F);
    stdDev_band_amp_ica_true  = zeros(nBands, param.N_F);
    stdDev_band_amp_ica_recon = zeros(nBands, param.N_F);
    
    for b = 1:nBands
        band = band_names{b};
        f_range = bands.(band);
        idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
        
        mean_band_amp_ica_true(b,:)  = mean(Ht_amp_ica(idx_band,:), 1, 'omitnan');
        mean_band_amp_ica_recon(b,:) = mean(Hr_amp_ica(idx_band,:), 1, 'omitnan');
        stdDev_band_amp_ica_true(b,:)  = std(Ht_amp_ica(idx_band,:), 0, 1, 'omitnan');
        stdDev_band_amp_ica_recon(b,:) = std(Hr_amp_ica(idx_band,:), 0, 1, 'omitnan');
    end
    
    true_vals_ica  = mean_band_amp_ica_true(:);
    recon_vals_ica = mean_band_amp_ica_recon(:);
    band_labels = repelem(band_names, param.N_F);
    
    fig3 = figure('Position',[50 50 1400 300]);
    tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['ICA True vs Reconstructed Band Mean FFT Amplitudes, (k=' num2str(num_comps) ')']);
    colors = lines(nBands);
    markers = {'o','s','d','h','^','hexagram'}; %,'hexagram','<','>'
    
    for b = 1:nBands    
        nexttile; hold on; 
        idx_b = strcmp(band_labels, band_names{b});
        x_ica = true_vals_ica(idx_b);
        y_ica = recon_vals_ica(idx_b);
        
        for m = 1:length(markers)
            if m > numel(x_ica), break; end
            scatter(x_ica(m), y_ica(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
            errorbar(x_ica(m), y_ica(m), stdDev_band_amp_ica_true(b, m), stdDev_band_amp_ica_recon(b, m), ...
                     'LineStyle', 'none', 'Color', colors(b,:), 'CapSize', 5,'HandleVisibility', 'off');
        end
        xfit_ica = linspace(min(x_ica), max(x_ica), 100);
        plot(xfit_ica, xfit_ica, 'Color', colors(b,:), 'LineWidth', 2, 'DisplayName', "y=x");
        R_fit_ica = corrcoef(x_ica, y_ica);
        R2_fit_ica = R_fit_ica(1,2)^2;
        text(mean(x_ica), mean(y_ica), sprintf('R^2=%.2f', R2_fit_ica), 'Color', colors(b,:), 'FontSize', 12)
        if b==1, xlabel('Mean True Amp.'); ylabel('Mean Recon. Amp.'); end
        title([band_names{b} ' band']); grid on;
    end
    
    % Proxy Legend
    nLatents = length(markers);
    proxy_handles = gobjects(nLatents + 1,1);
    for m = 1:nLatents
        proxy_handles(m) = scatter(nan, nan, 70, 'Marker', markers{m}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    end
    proxy_handles(end) = plot(nan, nan, 'k', 'LineWidth', 2);
    legend_labels = [arrayfun(@(m) sprintf('Z_{%s}', num2str(param.f_peak(m))), 1:length(markers), 'UniformOutput', false), {'y = x'}];
    legend(proxy_handles, legend_labels, 'Location','eastoutside','TextColor','k','IconColumnWidth',7, 'NumColumns',2);
    hold off;
    set(findall(gcf,'-property','FontSize'),'FontSize',14)
    saveas(fig3, fullfile(method_dir, ['ICA_Scatter_Mean' file_suffix '.png']));

    
    %% Plot 4: Scatter plot: True vs Reconstructed Band Amplitudes (per trial)  
    
    ica_R2_scores = plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, num_comps, "ICA", method_dir);
end
%% 6. Outputs
outICA = struct();
outICA.icasig_train = icasig_train;
outICA.icasig_test  = icasig_test;
outICA.h_recon_train   = h_rec_train;
outICA.h_recon_test   = h_rec_test;
outICA.MSE_train    = mean((h_test - h_rec_test).^2, 'all');
outICA.method_dir   = method_dir;
outICA.corr_ICA    = corr_ICA;
outICA.R_full = R_ICA; 
outICA.spectral_R2 = ica_R2_scores;  

% Close local figures
close All;

end