function [R2_test_avg, MSE_test_avg, outPCA] = runPCAAnalysis(eeg_train, eeg_test, h_train, h_test, param, k, method_dir)
% runPCAAnalysis Performs PCA and reconstruction for a SPECIFIC k components.
%
% Inputs:
%   eeg_train, eeg_test : Neural data (Channels x Time)
%   h_train, h_test     : True latent fields (Time x F)
%   param               : Structure with fields (.fs, .N_F, .f_peak)
%   k                   : The specific number of components to use
%   method_dir          : Directory string to save results

%% 1. Setup
if ~exist(method_dir, 'dir'), mkdir(method_dir); end
file_suffix = sprintf('_k%d', k);
h_f_colors = lines(param.N_F); 

%% 2. Run PCA
% We only calculate exactly k components
[coeff, score, ~, ~, explained] = pca(eeg_train', 'NumComponents', k);

% Project test data
score_test = (eeg_test' - mean(eeg_train', 1)) * coeff;

% Match components to latents (Limited to this k)
[corr_PCA, R_PCA] = match_components_to_latents(score, h_train, 'PCA', k);

%% 3. Reconstruction & Normalization
% Train regression weights: ICs * W = Latents
W = score(:, 1:k) \ h_train;

% Reconstruct
h_recon_train_raw = score(:, 1:k) * W;
h_recon_test_raw  = score_test(:, 1:k) * W;

% --- Normalization ---
% We normalize the reconstruction so std=1, matching the true latents
h_recon_train = h_recon_train_raw ./ std(h_recon_train_raw, 0, 1);
h_recon_test  = h_recon_test_raw  ./ std(h_recon_test_raw, 0, 1);

%% 4. Compute Performance Metrics (for this specific k)
R2_feat_test = zeros(1, param.N_F);
MSE_feat_test = zeros(1, param.N_F);

for f = 1:param.N_F
    % Test Metrics
    res_var_t = sum((h_test(:,f) - h_recon_test(:,f)).^2);
    tot_var_t = sum((h_test(:,f) - mean(h_test(:,f))).^2);
    
    R2_feat_test(f)  = 1 - (res_var_t / tot_var_t);
    MSE_feat_test(f) = mean((h_test(:,f) - h_recon_test(:,f)).^2);
end

% Averages to return to the main script's k-loop
R2_test_avg  = mean(R2_feat_test);
MSE_test_avg = mean(MSE_feat_test);

% Zero-Lag Correlation for plotting
zeroLagCorr_pca = zeros(1, param.N_F);
for f = 1:param.N_F
    c = corrcoef(h_test(:,f), h_recon_test(:,f));
    zeroLagCorr_pca(f) = c(1,2);
end

%% 5. Frequency Analysis (FFT Calculation)
% Using h_recon_test for FFT analysis
N = size(h_test, 1);
trial_dur = 1; 
L = round(trial_dur * param.fs);
nTrials = floor(N/L);
f_freq = (0:L-1)*(param.fs/L);
nHz = L/2 + 1;
f_plot = f_freq(1:nHz);
Ht = zeros(L, param.N_F, nTrials);
Hr = zeros(L, param.N_F, nTrials);
R2_trials = zeros(L, param.N_F, nTrials);

for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);
    Z_true_sub  = h_test(idx, :);
    Z_recon_sub = h_recon_test(idx, :);
    
    Ht(:,:,tr) = fft(Z_true_sub);
    Hr(:,:,tr) = fft(Z_recon_sub);
    
    for fidx = 1:param.N_F
        num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
        den = abs(Ht(:,fidx,tr)).^2 + eps;
        R2_trials(:,fidx,tr) = 1 - num./den;
    end
end
Ht_avg = mean(Ht, 3);
Hr_avg = mean(Hr, 3);
R2_avg = mean(R2_trials, 3);

% Band Averaging
bands = struct('delta', [1 4], 'theta', [4 8], 'alpha', [8 12], 'beta', [13 30], 'gamma', [30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);
band_avg_R2 = zeros(nBands, param.N_F);
for b = 1:nBands
    f_range = bands.(band_names{b});
    idx_b = f_freq >= f_range(1) & f_freq <= f_range(2);
    for fidx = 1:param.N_F
        band_avg_R2(b, fidx) = mean(R2_avg(idx_b, fidx));
    end
end

% ============================================================
% PLOTTING SECTION
% ============================================================
% Only plot if we aren't in a parallel worker (to save time)
if isempty(getCurrentTask()) && k > 4
    
    % Time Domain Plots
    plotTimeDomainReconstruction(h_test, h_recon_test, param, 'PCA', k, zeroLagCorr_pca, method_dir);
    
    % PC Traces
    plotCTraces(k, param, score, method_dir, file_suffix);
    
    % Cumulative Explained Variance (Scree Plot)
    fig_scree = figure('Position',[50 50 800 300]);
    plot(cumsum(explained), 'o-', 'LineWidth', 1.5);
    xlabel('Number of PCs'); ylabel('Cumulative Var (%)');
    title(sprintf('PCA Explained Variance (k=%d)', k));
    grid on;
    saveas(fig_scree, fullfile(method_dir, ['PCA_ScreePlot' file_suffix '.png']));
    close(fig_scree);
    
    % Frequency Analysis FFT
    fig_fft = figure('Position',[50 50 1000 600]);
    tiledlayout(2, 1);
    nexttile; loglog(f_plot, abs(Ht_avg(1:nHz,:))); title('True Latents FFT'); grid on;
    nexttile; loglog(f_plot, abs(Hr_avg(1:nHz,:))); title('Reconstructed FFT'); grid on;

    saveas(fig_fft, fullfile(method_dir, ['PCA_Frequency' file_suffix '.png']));
    close(fig_fft);

    % Bandwise R2 Bar Chart
    fig_band = figure('Position',[50 50 1000 300]);
    bar(band_avg_R2');
    set(gca, 'XTickLabel', arrayfun(@(i) sprintf('Z_{%s}', num2str(param.f_peak(i))), 1:param.N_F, 'UniformOutput', false));
    legend(band_names, 'Location', 'eastoutside');
    title(sprintf('Bandwise R^2 for k=%d', k));
    ylim([-1 1]); grid on;
    saveas(fig_band, fullfile(method_dir, ['PCA_Bandwise_R2' file_suffix '.png']));
    close(fig_band);
    
    % Band Amplitude Scatter
    plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, k, "PCA", method_dir);
end

%% 6. Final Output Structure
outPCA = struct();
outPCA.h_recon_train = h_recon_train; % Required for snippet logic
outPCA.h_recon_test  = h_recon_test;
outPCA.R2_features   = R2_feat_test;  % Per-latent R2
outPCA.MSE_features  = MSE_feat_test; % Per-latent MSE
outPCA.corr_PCA      = corr_PCA;      % Table of matches
outPCA.R_full        = R_PCA;         % Full corr matrix
outPCA.explained     = explained;
outPCA.method_dir    = method_dir;

close all;
end