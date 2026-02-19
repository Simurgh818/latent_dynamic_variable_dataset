function [R2_dpca_avg, MSE_dpca_avg, outDPCA] = rundPCAAnalysis( ...
        eeg_train, eeg_test, h_train, h_test, param, k, results_dir)
% rundPCAAnalysis: runs dPCA for a specific k, reconstructs latents, 
% and computes performance metrics.
%
% Inputs:
%   eeg_train             : nChannels x T
%   h_train    : T x N_F
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
num_f = size(h_train, 2);

%% 2. Run dPCA (Single Condition)
% Prepare X for dPCA: (Channels x Time x Trials) -> Trials=1
X_dpca = zeros(size(eeg_train,1), size(eeg_train,2), 1);
X_dpca(:,:,1) = eeg_train;

% Run dPCA for exactly 'k' components
% Note: If your dPCA library supports requesting specific k, this is fine.
% If it returns more, we slice it later.
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

%% ============================================================
%  PLOTTING SECTION
% ============================================================
% Only plot if running serially (main thread) and k is large enough
if isempty(getCurrentTask()) && k > 4
    
    % 1. Time Domain Reconstruction
    plotTimeDomainReconstruction(h_test, h_f_recon_normalized_dpca, ...
        param, 'dPCA', k, zeroLagCorr_dpca, results_dir);
    
    % 2. Component Traces
    plotCTraces(k, param, Z_dpca', results_dir, file_suffix);
    
    % 3. Explained Variance
    save_path = fullfile(results_dir, ['dPCA_ExplainedVariance' file_suffix '.png']);
    plotCumulativeVariance(explainedVar_pct, k, 'dPCA', save_path);

    % --- Frequency Analysis Prep ---
    save_path_fft = fullfile(results_dir, ['dPCA_FFT_True_vs_Recon' file_suffix '.png']);
    [outFSP] = plotFrequencySpectra(h_test, h_f_recon_normalized_dpca, 'dPCA', param, k, save_path_fft);

    Ht = outFSP.Ht;
    Hr = outFSP.Hr;
    Ht_avg = outFSP.Ht_avg;
    Hr_avg = outFSP.Hr_avg;
    R2_avg = outFSP.R2_avg;
    f_axis = outFSP.f_axis;
    f_plot = outFSP.f_plot;

    % 5. Band-wise R2 Bar Chart
    br2_path = fullfile(results_dir, ['dPCA_Bandwise_R2' file_suffix '.png']);
    [outBR2P] = plotBandwiseR2(R2_avg, f_axis, param, k, 'dPCA', br2_path);
    bands = outBR2P.bands;
    band_names = outBR2P.b_names; 

    % 6. Scatter Plots
    % (Keeping your scatter logic, simplified call)
    dpca_R2_scores = plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, k, "dPCA", results_dir);
end

%% 6. Package Output
outDPCA = struct();
outDPCA.W = W;
outDPCA.V = V;
outDPCA.Z_dpca = Z_dpca;
outDPCA.h_recon_test = h_f_recon_normalized_dpca; % Normalized reconstruction
outDPCA.zeroLagCorr = zeroLagCorr_dpca;
outDPCA.explainedVar_frac = explainedVar_frac;
outDPCA.explainedVar_pct  = explainedVar_pct;
outDPCA.folder = results_dir;
outDPCA.corr_dPCA = corr_dPCA;
outDPCA.R_full = R_dPCA;
outDPCA.R2_features = R2_feat;
outDPCA.MSE_features = MSE_feat;
outDPCA.spectral_R2 = dpca_R2_scores;   

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