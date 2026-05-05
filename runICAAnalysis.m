function [outICA] = runICAAnalysis(eeg_train, eeg_test, h_train, h_test, num_comps, param, method_dir)
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

% 1. Get the true mathematical limit
data_rank = rank(EEG.data); 
if num_comps > data_rank
    warning('Requested %d components, but data rank is %d. Capping components.', num_comps, data_rank);
    k = data_rank;
else
    k = num_comps;
end

% 2. Run ICA at FULL RANK
try
    [icasig, A, W] = fastica(eeg_train, ... 
        'numOfIC', k, ...
        'lastEig', k, ... 
        'verbose', 'off', ...
        'displayMode', 'off', ...
        'approach', 'symm', ...
        'g', 'tanh');
    
    icasig_train = icasig'; 
    
    % 3. Project Test Data
    train_mean = mean(eeg_train, 2);
    eeg_test_centered = eeg_test - train_mean;
    icasig_test = (W * eeg_test_centered)';
catch ME
    fprintf('FastICA failed or not installed. Falling back to MATLAB "rica". Error: %s \n', ME.message);
    Mdl = rica(eeg_train', k); 
    icasig_train = transform(Mdl, eeg_train');
    icasig_test = transform(Mdl, eeg_test');
end

% Mapping components to latents
C = icasig_test;   
H = h_test(1:size(C,1), :);
[corr_ICA, R_ICA] = match_components_to_latents(C, H, 'ICA', num_comps);

%% 3. Linear Mapping ICs -> Latent Fields
W_map = icasig_train \ h_train;

% Reconstruct Data (Raw)
h_rec_train_raw = icasig_train * W_map;
h_rec_test_raw  = icasig_test * W_map;

% --- NORMALIZATION STEP (MATCHING PCA/dPCA) ---
h_rec_train = h_rec_train_raw ./ std(h_rec_train_raw, 0, 1);
h_rec_test  = h_rec_test_raw  ./ std(h_rec_test_raw, 0, 1);

%% Calculate Metric

% Compute Zero-Lag Correlation for plotting
Corr_ica = zeros(1, param.N_F);
for f = 1:param.N_F
    c = corrcoef(h_test(:,f), h_rec_test(:,f));
    Corr_ica(f) = c(1,2);
end

%% 4. Frequency & Spectral R2 Math (Runs on ALL workers!)
T = size(h_test, 1);
trial_dur = 1; 
L = round(trial_dur * param.fs);
nTrials = floor(T/L);
f_axis = (0:L-1)*(param.fs/L);
nHz = floor(L/2) + 1;
f_plot = f_axis(1:nHz);

Ht = zeros(L, param.N_F, nTrials);
Hr = zeros(L, param.N_F, nTrials);
R2_trials = zeros(L, param.N_F, nTrials);

% --- FFT Calculation ---
for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);
    Ht(:,:,tr) = fft(h_test(idx, :));
    Hr(:,:,tr) = fft(h_rec_test(idx, :));
     for fidx = 1:param.N_F
        num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
        den = abs(Ht(:,fidx,tr)).^2 + eps;
        R2_trials(:,fidx,tr) = 1 - num./den;
     end
end

Ht_avg_ica = mean(abs(Ht(1:nHz, :, :)), 3);
Hr_avg_ica = mean(abs(Hr(1:nHz, :, :)), 3);
R2_avg_ica = mean(R2_trials, 3);

% --- Spectral R2 Calculation ---
bands = struct('delta',[1 4], 'theta',[4 8], 'alpha',[8 13], 'beta',[13 30], 'gamma',[30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);

ica_R2_scores = nan(param.N_F, 1);
Ht_amp = abs(Ht(1:nHz,:,:));
Hr_amp = abs(Hr(1:nHz,:,:));
max_t = max(Ht_amp(:)); if max_t==0, max_t=1; end
max_r = max(Hr_amp(:)); if max_r==0, max_r=1; end
Ht_amp = Ht_amp ./ max_t; Hr_amp = Hr_amp ./ max_r;

true_vals = cell(nBands,1);
recon_vals = cell(nBands,1);

for b = 1:nBands
    f_range  = bands.(band_names{b});
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
    
    true_vals{b}  = squeeze(mean(Ht_amp(idx_band,:,:), 1, 'omitnan'));
    recon_vals{b} = squeeze(mean(Hr_amp(idx_band,:,:), 1, 'omitnan'));
    
    if b == 4, target_zs = [4, 5];
    elseif b == 5, target_zs = 6;
    else, target_zs = b; end
    
    for z = 1:param.N_F
        if ismember(z, target_zs)
            x_z = true_vals{b}(z,:);
            y_z = recon_vals{b}(z,:);
            R_coef = corrcoef(x_z, y_z);
            if numel(R_coef) > 1, r_sq = R_coef(1,2)^2; else, r_sq = 0; end
            ica_R2_scores(z) = r_sq;
        end
    end
end

%% ============================================================
% PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if isempty(getCurrentTask())
    % Time domain plot
    plotTimeDomainReconstruction(h_test, h_rec_test, param, 'ICA', num_comps, Corr_ica, method_dir);
    
    % Independent Component traces plot
    plotCTraces(num_comps, param, h_rec_test, method_dir, file_suffix);
    
    % Frequency Analysis FFT (Simplified call)
    save_path_fft = fullfile(method_dir, ['ICA_FFT_True_vs_Recon' file_suffix '.png']);
    plotFrequencySpectra(Ht_avg_ica, Hr_avg_ica, f_plot, 'ICA', param, num_comps, save_path_fft);
 
    %% Plot 2: Band-wise R2 Bar Chart
    br2_path = fullfile(method_dir, ['ICA_Bandwise_R2' file_suffix '.png']);
    plotBandwiseR2(R2_avg_ica, f_axis, param, num_comps, 'ICA', br2_path);

    %% Plot 3: Scatter plot: True vs Reconstructed Band Amplitudes (Mean)
    % (Keeping your original manual scatter plot logic for the Means)
    Ht_amp_ica_norm = Ht_avg_ica(1:nHz, :) ./ max(Ht_avg_ica(1:nHz, :), [], 'all');
    Hr_amp_ica_norm = Hr_avg_ica(1:nHz, :) ./ max(Hr_avg_ica(1:nHz, :), [], 'all');
    
    mean_band_amp_ica_true  = zeros(nBands, param.N_F);
    mean_band_amp_ica_recon = zeros(nBands, param.N_F);
    stdDev_band_amp_ica_true  = zeros(nBands, param.N_F);
    stdDev_band_amp_ica_recon = zeros(nBands, param.N_F);
    
    for b = 1:nBands
        band = band_names{b};
        f_range = bands.(band);
        idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
        
        mean_band_amp_ica_true(b,:)  = mean(Ht_amp_ica_norm(idx_band,:), 1, 'omitnan');
        mean_band_amp_ica_recon(b,:) = mean(Hr_amp_ica_norm(idx_band,:), 1, 'omitnan');
        stdDev_band_amp_ica_true(b,:)  = std(Ht_amp_ica_norm(idx_band,:), 0, 1, 'omitnan');
        stdDev_band_amp_ica_recon(b,:) = std(Hr_amp_ica_norm(idx_band,:), 0, 1, 'omitnan');
    end
    
    true_vals_ica  = mean_band_amp_ica_true(:);
    recon_vals_ica = mean_band_amp_ica_recon(:);
    band_labels = repelem(band_names, param.N_F);
    
    fig3 = figure('Position',[50 50 1400 300], 'Visible', 'off');
    tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['ICA True vs Reconstructed Band Mean FFT Amplitudes, (k=' num2str(num_comps) ')']);
    colors = lines(nBands);
    markers = {'o','s','d','h','^','hexagram'};
    
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
    
    nLatents = length(markers);
    proxy_handles = gobjects(nLatents + 1,1);
    for m = 1:nLatents
        proxy_handles(m) = scatter(nan, nan, 70, 'Marker', markers{m}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    end
    proxy_handles(end) = plot(nan, nan, 'k', 'LineWidth', 2);
    legend_labels = [arrayfun(@(m) sprintf('Z_{%s}', num2str(param.f_peak(m))), 1:length(markers), 'UniformOutput', false), {'y = x'}];
    legend(proxy_handles, legend_labels, 'Location','eastoutside','TextColor','k','IconColumnWidth',7, 'NumColumns',2);
    hold off;
    set(findall(fig3,'-property','FontSize'),'FontSize',14)
    saveas(fig3, fullfile(method_dir, ['ICA_Scatter_Mean' file_suffix '.png']));
    close(fig3);
    
    %% Plot 4: Scatter plot: True vs Reconstructed Band Amplitudes (per trial)  
    plotBandScatterPerTrial(true_vals, recon_vals, ica_R2_scores, band_names, param, num_comps, "ICA", method_dir);
end

%% 6. Outputs
outICA = struct();
outICA.icasig_train = icasig_train;
outICA.icasig_test  = icasig_test;
outICA.h_recon_train = h_rec_train;
outICA.h_recon_test  = h_rec_test;
outICA.method_dir   = method_dir;
outICA.corr_ICA     = corr_ICA;
outICA.R_full       = R_ICA; 
outICA.Corr  = Corr_ica;
outICA.spectral_R2  = ica_R2_scores;  % <--- Now guaranteed to exist!

close all;
end