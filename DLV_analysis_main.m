function DLV_analysis_main(param)
% Modular functions for PCA, ICA, and UMAP analyses of DLV model data
% Main wrapper to run PCA, ICA, and UMAP analyses
% Inputs:
%   s_i   : N_neur x T binary spike matrix
%   param : struct with fields (N_neur, N_F, ... )

[s_i, param, h_f] = sampleMorrellModel(param);

% testing Spike to EEG function:
target_bin_size = 0.05; % 50 ms
tau = 25; % Post-synaptic kernel, Alpha kernel's number of bins to the peak of the kernel
smooth_sigma = 5; % Std. Dev. of the Gaussian kernel in bins
group_size = 8; % grouping of neurons per channel

[s_eeg_like, h_f_processed] = spike_to_eeg(s_i, h_f, param, target_bin_size, tau, group_size, smooth_sigma);
% Marchenko–Pastur threshold
eig_vals = eig(cov(s_eeg_like'));
[N, T] = size(s_eeg_like);
Q = T / N;
sigma2 = mean(eig_vals);           % crude estimate of noise variance
lambda_max = sigma2 * (1 + sqrt(1/Q))^2;

% Find number of components above the MP threshold
num_sig_components = sum(eig_vals > lambda_max);
fprintf('Marchenko–Pastur suggests keeping %d PCs\n', num_sig_components);

[s_i_test, param_test, h_f_test] = sampleMorrellModel(param);
[s_eeg_like_test, h_f_processed_test] = spike_to_eeg(s_i_test,h_f_test, param, target_bin_size,tau, group_size, smooth_sigma);


% 1. PCA Analysis (train & test)
[coeff, score, explained, rec_err_pca] = runPCAAnalysis(s_eeg_like, s_eeg_like_test, h_f_processed, h_f_processed_test, param, num_sig_components);

% 2. ICA Analysis (train & test)
[icasig, rec_err] = runICAAnalysis(s_eeg_like, s_eeg_like_test, h_f_processed, h_f_processed_test, param, num_sig_components);

n_neighbors = 10;
min_dist    = 0.10;

% 3. UMAP Analysis (train & test clusters)
[umap_s_i, umap_s_i_test, reconstruction_error_umap, reconstruction_error_test_umap] = ...
    runUMAPAnalysis(n_neighbors, min_dist, s_eeg_like, s_eeg_like_test, param, h_f_processed, h_f_processed_test, num_sig_components);


end