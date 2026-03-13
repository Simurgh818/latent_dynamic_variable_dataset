function [R2_test, MSE_test, outAE] = runAutoencoderAnalysis(X_train, X_test, H_train, H_test, bottleNeck, param,  results_dir)
% runAutoencoderAnalysis Trains AE + Linear Mapping and generates diagnostic plots
%
% Inputs:
%   X_train, X_test     : Neural data (Neurons x Time or Time x Neurons depending on your AE setup)
%   H_train, H_test     : True latent fields (Time x F)
%   bottleNeck          : Int, size of the bottleneck layer (k)
%   param               : Structure with fields .N_F, .f_peak
%   results_dir         : Directory to save results
%
% Outputs:
%   R2_test, MSE_test   : Global performance metrics on Test set
%   outAE               : Structure containing networks, traces, and metrics

%% 1. Setup and Directory
method_name = 'AE';
method_dir = fullfile(results_dir, method_name);
if ~exist(method_dir, 'dir')
    mkdir(method_dir);
end
file_suffix = sprintf('_k%d', bottleNeck);
h_f_colors = lines(param.N_F); 
fs_new = param.fs;

%% 2. Train Autoencoder (Unsupervised)
batch_size = 100; % 80

[net, ~] = trainEEGAutoencoder(X_train, X_test,  ...
    'encoderLayerSizes', [64,32], ...
    'bottleneckSize', bottleNeck, ...
    'decoderLayerSizes', [32,64], ...
    'encoderActivations', {'tanh','tanh'}, ...
    'decoderActivations', {'tanh','tanh'}, ...
    'outputActivation', "none", ...
    'epochs', 50, ...
    'batchSize', batch_size, ...
    'learnRate', 1e-3);

% Extract Latents
Z_train_c = double(activations(net, X_train.', 'bottleneck', 'OutputAs','rows'));
Z_test_c  = double(activations(net, X_test.',  'bottleneck', 'OutputAs','rows'));
H_train   = double(H_train);
H_test    = double(H_test);

% Ensure matching lengths
minLen = min(size(Z_train_c,1), size(H_train,1));
Z_train_c = Z_train_c(1:minLen,:);
H_train   = H_train(1:minLen,:);
minLenTest = min(size(Z_test_c,1), size(H_test,1));
Z_test_c  = Z_test_c(1:minLenTest,:);
H_test    = H_test(1:minLenTest,:);

% Matching components to latents
[corr_AE, R_AE] = match_components_to_latents(Z_test_c, H_test, 'AE', bottleNeck);

%% 3. Linear Mapping via lsqlin / matrix division
H_recon_train = zeros(size(H_train));
H_recon_test  = zeros(size(H_test));
recon_R2_train = zeros(1,param.N_F);
recon_R2_test  = zeros(1,param.N_F);

for f = 1:param.N_F
    w = Z_train_c \ H_train(:,f);
    
    % Reconstruct
    H_recon_train(:,f) = Z_train_c * w;
    H_recon_test(:,f)  = Z_test_c * w;
    
    % Component-wise R^2
    recon_R2_train(f) = 1 - sum((H_train(:,f) - H_recon_train(:,f)).^2) / sum((H_train(:,f) - mean(H_train(:,f))).^2);
    recon_R2_test(f)  = 1 - sum((H_test(:,f) - H_recon_test(:,f)).^2) / sum((H_test(:,f) - mean(H_test(:,f))).^2);
end

%% 4. Global Metrics & Zero-Lag Correlation
R2_test  = 1 - sum((H_test(:) - H_recon_test(:)).^2) / sum((H_test(:) - mean(H_test(:))).^2);
MSE_test = mean((H_test(:) - H_recon_test(:)).^2);

Z_true_train = H_train;
Z_true_test = H_test;
Z_recon_train = H_recon_train;
Z_recon_test = H_recon_test;

maxLag = 200;               
lags   = -maxLag:maxLag;    

zeroLagCorr_train = zeros(1, param.N_F);
zeroLagCorr_test = zeros(1, param.N_F);
for f = 1:param.N_F
    c_train = xcorr(Z_true_train(:,f), Z_recon_train(:,f), maxLag, 'coeff');
    c_test = xcorr(Z_true_test(:,f), Z_recon_test(:,f), maxLag, 'coeff');
    zeroLagCorr_train(f) = c_train(lags==0);
    zeroLagCorr_test(f) = c_test(lags==0);
end

%% 5. Frequency & Spectral R2 Math (Runs on ALL workers!)
T = size(H_test, 1);
trial_dur = 1; 
L = round(trial_dur * param.fs);
nTrials = floor(T/L);
f_axis = (0:L-1)*(param.fs/L);
nHz = floor(L/2) + 1;
f_plot = f_axis(1:nHz);

Ht_ae = zeros(L, param.N_F, nTrials);
Hr_ae = zeros(L, param.N_F, nTrials);
R2_trials = zeros(L, param.N_F, nTrials);

% --- FFT Calculation ---
for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);
    Ht_ae(:,:,tr) = fft(H_test(idx, :));
    Hr_ae(:,:,tr) = fft(H_recon_test(idx, :));
     for fidx = 1:param.N_F
        num = abs(Ht_ae(:,fidx,tr) - Hr_ae(:,fidx,tr)).^2;
        den = abs(Ht_ae(:,fidx,tr)).^2 + eps;
        R2_trials(:,fidx,tr) = 1 - num./den;
     end
end

Ht_avg_ae = mean(abs(Ht_ae(1:nHz, :, :)), 3);
Hr_avg_ae = mean(abs(Hr_ae(1:nHz, :, :)), 3);
R2_avg_ae = mean(R2_trials, 3);

% --- Spectral R2 Calculation ---
bands = struct('delta',[1 4], 'theta',[4 8], 'alpha',[8 13], 'beta',[13 30], 'gamma',[30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);

AE_R2_scores = nan(param.N_F, 1);
Ht_amp = abs(Ht_ae(1:nHz,:,:));
Hr_amp = abs(Hr_ae(1:nHz,:,:));
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
            AE_R2_scores(z) = r_sq;
        end
    end
end


%% ============================================================
% PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if isempty(getCurrentTask()) 

    %% Plot 1 & 2: Component Traces
    plotCTraces(bottleNeck, param, H_recon_test, method_dir, file_suffix);
    
    %% Plot 3: Detailed Reconstruction (Train & Test Split)
    fig3 = figure('Name','True vs. AE Latents', 'Position', [50, 100, 1500, 150*param.N_F], 'Visible', 'off');
    tiledlayout(param.N_F, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['AE (k=' num2str(bottleNeck) ') True Latents (solid) vs. Reconstructed (dashed)'],'FontSize',30);
    
    vis_len_train = min(fs_new*2, size(H_train,1)); 
    vis_len_test = min(fs_new*2, size(H_test,1));
    
    for f = 1:param.N_F
        % --- Train Column ---
        nexttile; hold on;
        set(gca, 'XColor', 'none', 'YColor', 'none'); box on
        plot(H_train(1:vis_len_train, f), '-',  'Color', h_f_colors(f,:), 'LineWidth', 1.5,'DisplayName', ['$Z_{' num2str(param.f_peak(f)) '}$']); 
        plot(H_recon_train(1:vis_len_train, f), '--', 'Color', 'k', 'LineWidth', 1.5,'DisplayName', ['$\hat{Z}_{' num2str(param.f_peak(f)) '}$']);
        xlim([0 fs_new]);
        if f==1, title('Training Set'); end
        legend('location', 'southeastoutside', 'Interpreter', 'latex');
        rho_train = zeroLagCorr_train(f);
        text(0.02 * fs_new, 0.05 * max(H_train(:,f)), ...
        sprintf('\\rho(0)=%.2f', rho_train), ...
        'FontSize', 12, 'FontWeight', 'bold',...
        'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
        'Margin', 3, 'EdgeColor','k');
    
        % --- Test Column ---
        nexttile; hold on;
        set(gca, 'XColor', 'none', 'YColor', 'none'); box on
        plot(H_test(1:vis_len_test, f), '-',  'Color', h_f_colors(f,:), 'LineWidth', 1.5, 'DisplayName',[' $Z_{' num2str(param.f_peak(f)) '}$']);  
        plot(H_recon_test(1:vis_len_test, f), '--', 'Color', 'k', 'LineWidth', 1.5, 'DisplayName', ['$\hat{Z}_{' num2str(param.f_peak(f)) '}$']);
        xlim([0 fs_new]);
        rho_test = zeroLagCorr_test(f);
        text(0.02 * fs_new, 0.05 * max(H_test(:,f)), ...
        sprintf('\\rho(0)=%.2f', rho_test), ...
        'FontSize', 12, 'FontWeight', 'bold',...
        'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
        'Margin', 3, 'EdgeColor','k');
        legend('location', 'southeastoutside', 'Interpreter', 'latex');
        if f==1, title('Test Set'); end
    end
    set(findall(fig3,'-property','FontSize'),'FontSize',28);
    saveas(fig3, fullfile(method_dir, ['AE_Split_Reconstruction' file_suffix '.png']));  
    close(fig3);
    
    %% Plot 4: Frequency Analysis FFT
    save_path_fft = fullfile(method_dir, ['AE_FFT_True_vs_Recon' file_suffix '.png']);
    plotFrequencySpectra(Ht_avg_ae, Hr_avg_ae, f_plot, 'AE', param, bottleNeck, save_path_fft);
    
    %% Plot 5: Band R2
    br2_path = fullfile(method_dir, ['AE_Bandwise_R2' file_suffix '.png']);
    plotBandwiseR2(R2_avg_ae, f_axis, param, bottleNeck, 'AE', br2_path);
    
    %% Plot 6: Band Power Bar Chart (Using original Ht/Hr for un-normalized power)
    band_power_true = zeros(nBands, param.N_F);
    band_power_recon = zeros(nBands, param.N_F);
    band_power_true_std = zeros(nBands, param.N_F);
    band_power_recon_std = zeros(nBands, param.N_F);
    
    for b = 1:nBands
        f_range = bands.(band_names{b});
        idx = f_plot >= f_range(1) & f_plot <= f_range(2);
        for fidx = 1:param.N_F
            trial_power_true = mean(abs(Ht_ae(idx,fidx,:)).^2, 1, 'omitnan');
            trial_power_recon = mean(abs(Hr_ae(idx,fidx,:)).^2, 1, 'omitnan');
            
            band_power_true(b,fidx) = mean(trial_power_true(:));
            band_power_recon(b,fidx) = mean(trial_power_recon(:));
            band_power_true_std(b,fidx) = std(trial_power_true(:));
            band_power_recon_std(b,fidx) = std(trial_power_recon(:));
        end
    end
    
    fig5 = figure('Position',[100 100 1000 250*ceil(param.N_F/3)], 'Visible', 'off');
    tiledlayout(ceil(param.N_F/3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['Band Power Comparison (Mean ± SD) for k = ' num2str(bottleNeck)]);
    
    for fidx = 1:param.N_F
        nexttile;
        bar_data = [band_power_true(:,fidx), band_power_recon(:,fidx)];
        bar_std = [band_power_true_std(:,fidx), band_power_recon_std(:,fidx)];
        bh = bar(bar_data); hold on;
        
        ngroups = size(bar_data,1); nbars = size(bar_data,2);
        groupwidth = min(0.8, nbars/(nbars + 1.5));
        for i = 1:nbars
            x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
            errorbar(x, bar_data(:,i), bar_std(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1);
        end
        bh(1).FaceColor = [0.3 0.6 0.9]; bh(2).FaceColor = [0.9 0.4 0.4];
        set(gca, 'XTickLabel', band_names, 'XTickLabelRotation', 45);
        title(['Z_{' num2str(param.f_peak(fidx)) '}']); 
        grid on;
    end
    legend({'True','Reconstructed'}, 'Location','northoutside','Orientation','horizontal');
    set(findall(fig5,'-property','FontSize'),'FontSize',16);
    saveas(fig5, fullfile(method_dir, ['AE_Band_Power' file_suffix '.png']));
    close(fig5);

    %% Plot 7: Scatter Mean Band Amplitudes (Simplified using true_vals)
    fig7 = figure('Position',[50 50 1200 300], 'Visible', 'off');
    tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['AE: True vs Reconstructed Band Mean FFT Amplitudes for k= ' num2str(bottleNeck)]);
    colors = lines(nBands);
    markers = {'o','s','d','h','^','hexagram'}; 
    
    for b = 1:nBands    
        nexttile; hold on;
        
        % We can reuse the true_vals and recon_vals generated in Section 5
        x = mean(true_vals{b}, 2, 'omitnan'); % Mean across trials
        y = mean(recon_vals{b}, 2, 'omitnan');
        
        % Std across trials
        std_x = std(true_vals{b}, 0, 2, 'omitnan');
        std_y = std(recon_vals{b}, 0, 2, 'omitnan');

        for m = 1:length(markers)
            if m > numel(x), break; end
            scatter(x(m), y(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
            errorbar(x(m), y(m), std_x(m), std_y(m), 'LineStyle', 'none', 'Color', colors(b,:), 'CapSize', 5);
        end
        plot(linspace(min(x),max(x)), linspace(min(x),max(x)), 'Color', colors(b,:), 'LineWidth', 2);
        R = corrcoef(x, y);
        text(mean(x), mean(y), sprintf('R^2=%.2f', R(1,2)^2), 'Color', colors(b,:), 'FontSize', 12);
        title([band_names{b} ' band']); grid on;
    end
    set(findall(fig7,'-property','FontSize'),'FontSize',16);
    saveas(fig7, fullfile(method_dir, ['AE_Scatter_Mean' file_suffix '.png']));
    close(fig7);
    
    %% Plot 8: Scatter Per-Trial Band Amplitudes
    plotBandScatterPerTrial(true_vals, recon_vals, AE_R2_scores, band_names, param, bottleNeck, "AE", method_dir);
end

%% 6. Outputs and Summary Saves
outAE = struct();
outAE.net = net;
outAE.h_recon_train     = H_recon_train;
outAE.h_recon_test      = H_recon_test;
outAE.R2_test          = R2_test;
outAE.MSE_test         = MSE_test;
outAE.component_R2     = recon_R2_test;
outAE.results_dir      = method_dir;
outAE.corr_AE          = corr_AE;
outAE.R_full           = R_AE;
outAE.zeroLagCorr      = zeroLagCorr_test;
outAE.spectral_R2      = AE_R2_scores;  % Guaranteed to exist for parallel workers!

close all;

end