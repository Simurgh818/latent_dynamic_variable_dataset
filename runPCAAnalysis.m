function [outPCA] = runPCAAnalysis(eeg_train, eeg_test, h_train, h_test, param, k, method_dir)
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

% Ensure we only pass the first 'k' components
score_train_k = score(:, 1:k);
score_test_k  = score_test(:, 1:k);

[h_recon_train, h_recon_test, Comp_latent_matching_corr_pca, R_PCA, direct_Component_Corr_pca, pca_R2_scores, freq_data] = ...
    computePerformanceMetrics(score_train_k, score_test_k, h_train, h_test, 'PCA', k, param);

%% ============================================================
% PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if (isempty(getCurrentTask()) & k==10)
    plotTimeDomainReconstruction(h_test, h_recon_test, param, 'PCA', k, direct_Component_Corr_pca, method_dir);
    plotCTraces(k, param, score_test, method_dir, file_suffix);
    
    save_path = fullfile(method_dir, ['PCA_ExplainedVariance' file_suffix '.png']);
    plotCumulativeVariance(explained, k, 'PCA', save_path);
    
    % Pass pre-calculated math into simplified plotting functions
    save_path_fft = fullfile(method_dir, ['PCA_FFT_True_vs_Recon' file_suffix '.png']);
    % FFT Spectra
    plotFrequencySpectra(freq_data.Ht_avg, freq_data.Hr_avg, freq_data.f_plot, 'PCA', param, k, save_path_fft);
        
    br2_path = fullfile(method_dir, ['PCA_Bandwise_R2' file_suffix '.png']);
    % Bandwise R2
    plotBandwiseR2(freq_data.R2_avg, freq_data.f_axis, param, k, 'PCA', br2_path);
    % Scatter per trial
    plotBandScatterPerTrial(freq_data.true_vals, freq_data.recon_vals, pca_R2_scores, freq_data.band_names, param, k, "PCA", method_dir);
end

%% 6. Final Output Structure
outPCA = struct();
outPCA.h_recon_train = h_recon_train; % Required for snippet logic
outPCA.h_recon_test  = h_recon_test;
outPCA.matched_R2       = freq_data.matched_R2;
outPCA.Comp_latent_matching_corr      = Comp_latent_matching_corr_pca;      % Table of matches
outPCA.Comp_latent_matching_matrix        = R_PCA;         % Full corr matrix
outPCA.explained     = explained;
outPCA.method_dir    = method_dir;
outPCA.direct_Component_Corr = direct_Component_Corr_pca; 
outPCA.spectral_R2 = pca_R2_scores;    

close all;
end