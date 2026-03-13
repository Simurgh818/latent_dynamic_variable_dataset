function [R2_dpca_avg, MSE_dpca_avg, outDPCA] = rundPCAAnalysis( ...
        eeg_train, eeg_test, h_train, h_test, param, k, results_dir)
% rundPCAAnalysis: runs dPCA for a specific k, reconstructs latents, 
% and computes performance metrics.
%
% Inputs:
%   eeg_train            : nChannels x T
%   h_train              : T x N_F
%   param                : struct (uses param.f_peak)
%   k                    : scalar (Number of components to use)
%   results_dir          : directory to save figures
%
% Outputs:
%   R2_dpca_avg          : Scalar average R^2 across all latents
%   MSE_dpca_avg         : Scalar average MSE across all latents
%   outDPCA              : Structure with detailed results

%% 1. Setup
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
file_suffix = sprintf('_k%d', k);
num_f = size(h_test, 2);

%% 2. Run dPCA (Single Condition)
% Prepare X for dPCA: (Channels x Time x Trials) -> Trials=1
X_dpca = zeros(size(eeg_train,1), size(eeg_train,2), 1);
X_dpca(:,:,1) = eeg_train;

% Run dPCA for exactly 'k' components
[W, V, whichMarg] = dpca(X_dpca, k); 

% Latent time series (Components x Time)
Z_dpca = W' * eeg_test;         
Z_dpca_T = Z_dpca';             % T x nComp

% Match components to latents
H = h_test(1:size(Z_dpca_T,1), :);
[corr_dPCA, R_dPCA] = match_components_to_latents(Z_dpca_T, H, 'dPCA', k);

%% 3. Reconstruction & Metrics (No inner loop!)
% We use ALL k components to reconstruct the latents
h_f_recon_dpca = zeros(size(h_test));
h_f_recon_normalized_dpca = zeros(size(h_test));
R2_feat  = zeros(1, num_f);
MSE_feat = zeros(1, num_f);

for f = 1:num_f
    % Linear regression: Z_dpca_T * w = h_f
    % We use all columns (1:k) of Z_dpca_T
    w = Z_dpca_T \ h_test(:,f);
    
    % Reconstruction
    rec_raw = Z_dpca_T * w;
    h_f_recon_dpca(:,f) = rec_raw;
    
    % --- NORMALIZATION STEP ---
    % Normalize so std=1, matching the ground truth
    rec_norm = rec_raw ./ std(rec_raw);
    h_f_recon_normalized_dpca(:,f) = rec_norm;
    
    % Compute Metrics
    res_var = sum((h_test(:,f) - rec_norm).^2);
    tot_var = sum((h_test(:,f) - mean(h_test(:,f))).^2);
    
    R2_feat(f)  = 1 - (res_var / tot_var);
    MSE_feat(f) = mean((h_test(:,f) - rec_norm).^2);
end

% Averages to return to main script
R2_dpca_avg  = mean(R2_feat);
MSE_dpca_avg = mean(MSE_feat);

%% 4. Zero-lag correlation (for plotting)
maxLag = 200;
lags   = -maxLag:maxLag;
zeroLagCorr_dpca = zeros(1, num_f);
for f = 1:num_f
    c = xcorr(h_test(:,f), h_f_recon_normalized_dpca(:,f), maxLag, 'coeff');
    zeroLagCorr_dpca(f) = c(lags==0);
end

%% 5. Explained Variance
[explainedVar_frac, explainedVar_pct, explainedVar_cum] = ...
    dpca_explained_variance(X_dpca, W, V);

%% 6. Frequency & Spectral R2 Math (Runs on ALL workers!)
T = size(h_test, 1);
trial_dur = 1; 
L = round(trial_dur * param.fs);
nTrials = floor(T/L);
f_axis = (0:L-1)*(param.fs/L);
nHz = floor(L/2) + 1;
f_plot = f_axis(1:nHz);

Ht = zeros(L, num_f, nTrials);
Hr = zeros(L, num_f, nTrials);
R2_trials = zeros(L, num_f, nTrials);

% --- FFT Calculation ---
for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);
    Ht(:,:,tr) = fft(h_test(idx, :));
    Hr(:,:,tr) = fft(h_f_recon_normalized_dpca(idx, :));
     for fidx = 1:num_f
        num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
        den = abs(Ht(:,fidx,tr)).^2 + eps;
        R2_trials(:,fidx,tr) = 1 - num./den;
     end
end

Ht_avg = mean(abs(Ht(1:nHz, :, :)), 3);
Hr_avg = mean(abs(Hr(1:nHz, :, :)), 3);
R2_avg = mean(R2_trials, 3);

% --- Spectral R2 Calculation ---
bands = struct('delta',[1 4], 'theta',[4 8], 'alpha',[8 13], 'beta',[13 30], 'gamma',[30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);

dpca_R2_scores = nan(num_f, 1);
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
            dpca_R2_scores(z) = r_sq;
        end
    end
end

%% ============================================================
%  PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if isempty(getCurrentTask())
    
    % 1. Time Domain Reconstruction
    plotTimeDomainReconstruction(h_test, h_f_recon_normalized_dpca, ...
        param, 'dPCA', k, zeroLagCorr_dpca, results_dir);
    
    % 2. Component Traces
    plotCTraces(k, param, Z_dpca', results_dir, file_suffix);
    
    % 3. Explained Variance
    save_path = fullfile(results_dir, ['dPCA_ExplainedVariance' file_suffix '.png']);
    plotCumulativeVariance(explainedVar_pct, k, 'dPCA', save_path);
    
    % 4. Frequency Analysis FFT
    save_path_fft = fullfile(results_dir, ['dPCA_FFT_True_vs_Recon' file_suffix '.png']);
    plotFrequencySpectra(Ht_avg, Hr_avg, f_plot, 'dPCA', param, k, save_path_fft);
    
    % 5. Band-wise R2 Bar Chart
    br2_path = fullfile(results_dir, ['dPCA_Bandwise_R2' file_suffix '.png']);
    plotBandwiseR2(R2_avg, f_axis, param, k, 'dPCA', br2_path);
    
    % 6. Scatter Plots
    plotBandScatterPerTrial(true_vals, recon_vals, dpca_R2_scores, band_names, param, k, "dPCA", results_dir);
end

%% 7. Package Output
outDPCA = struct();
outDPCA.W = W;
outDPCA.V = V;
outDPCA.Z_dpca = Z_dpca;
outDPCA.h_recon_test = h_f_recon_normalized_dpca; 
outDPCA.zeroLagCorr = zeroLagCorr_dpca;
outDPCA.explainedVar_frac = explainedVar_frac;
outDPCA.explainedVar_pct  = explainedVar_pct;
outDPCA.folder = results_dir;
outDPCA.corr_dPCA = corr_dPCA;
outDPCA.R_full = R_dPCA;
outDPCA.R2_features = R2_feat;
outDPCA.MSE_features = MSE_feat;
outDPCA.zeroLagCorr = zeroLagCorr_dpca;
outDPCA.spectral_R2 = dpca_R2_scores; % <--- Now guaranteed to exist!
   
close all;
end

%% Nested Helper Function
function [explainedVar_frac, explainedVar_pct, explainedVar_cum] = dpca_explained_variance(X, W, V)
    % Ensure shaped as n x (T*C)
    if ismatrix(X)
        Xflat = X;                    
    elseif ndims(X) == 3
        Xflat = reshape(X, size(X,1), []); 
    else
        error('X must be 2D or 3D array');
    end
    % Mean-center 
    Xmean = mean(Xflat, 2);           
    Xc = bsxfun(@minus, Xflat, Xmean);
    % Total variance 
    perChannelVar = var(Xc, 0, 2);    
    totalVar = sum(perChannelVar(:)) + eps;
    % Project to components 
    Z = V' * Xc;                       
    nComp = size(Z,1);
    explainedVar_frac = zeros(1,nComp);
    % Compute contribution 
    for c = 1:nComp
        Recon_c = W(:,c) * Z(c,:);    
        explainedVar_frac(c) = sum(var(Recon_c, 0, 2)) / totalVar;
    end
    % Normalize and convert to percent
    explainedVar_frac = explainedVar_frac ./ sum(explainedVar_frac); 
    explainedVar_pct  = 100 * explainedVar_frac;
    explainedVar_cum  = cumsum(explainedVar_pct);
end