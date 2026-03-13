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
[corr_PCA, R_PCA] = match_components_to_latents(score_test, h_test, 'PCA', k);

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

%% 5. Frequency & Spectral R2 Math (Runs on ALL workers!)
T = size(h_test, 1);
num_f = size(h_test, 2);
trial_dur = 1; 
L = round(trial_dur * param.fs);
nTrials = floor(T/L);
f_axis = (0:L-1)*(param.fs/L);
nHz = floor(L/2) + 1;
f_plot = f_axis(1:nHz);

Ht = zeros(L, num_f, nTrials);
Hr = zeros(L, num_f, nTrials);
R2_trials = zeros(L, param.N_F, nTrials);

% --- FFT Calculation ---
for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);
    Ht(:,:,tr) = fft(h_test(idx, :));
    Hr(:,:,tr) = fft(h_recon_test(idx, :));
     for fidx = 1:param.N_F
        num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
        den = abs(Ht(:,fidx,tr)).^2 + eps;
        R2_trials(:,fidx,tr) = 1 - num./den;
     end
end

Ht_avg = mean(abs(Ht(1:nHz, :, :)), 3);
Hr_avg = mean(abs(Hr(1:nHz, :, :)), 3);
R2_avg = mean(R2_trials, 3);

% --- Spectral R2 Calculation (Band Amplitude Scatter Logic) ---
bands = struct('delta',[1 4], 'theta',[4 8], 'alpha',[8 13], 'beta',[13 30], 'gamma',[30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);

pca_R2_scores = nan(num_f, 1);
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
    
    for z = 1:num_f
        if ismember(z, target_zs)
            x_z = true_vals{b}(z,:);
            y_z = recon_vals{b}(z,:);
            R_coef = corrcoef(x_z, y_z);
            if numel(R_coef) > 1, r_sq = R_coef(1,2)^2; else, r_sq = 0; end
            pca_R2_scores(z) = r_sq;
        end
    end
end
%% ============================================================
% PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if isempty(getCurrentTask()) 
    plotTimeDomainReconstruction(h_test, h_recon_test, param, 'PCA', k, zeroLagCorr_pca, method_dir);
    plotCTraces(k, param, score_test, method_dir, file_suffix);
    
    save_path = fullfile(method_dir, ['PCA_ExplainedVariance' file_suffix '.png']);
    plotCumulativeVariance(explained, k, 'PCA', save_path);
    
    % Pass pre-calculated math into simplified plotting functions
    save_path_fft = fullfile(method_dir, ['PCA_FFT_True_vs_Recon' file_suffix '.png']);
    plotFrequencySpectra(Ht_avg, Hr_avg, f_plot, 'PCA', param, k, save_path_fft);
    
    br2_path = fullfile(method_dir, ['PCA_Bandwise_R2' file_suffix '.png']);
    plotBandwiseR2(R2_avg, f_axis, param, k, 'PCA', br2_path);
    
    plotBandScatterPerTrial(true_vals, recon_vals, pca_R2_scores, band_names, param, k, "PCA", method_dir);
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
outPCA.zeroLagCorr = zeroLagCorr_pca;
outPCA.spectral_R2 = pca_R2_scores;    

close all;
end