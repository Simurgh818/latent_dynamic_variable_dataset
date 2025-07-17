function [coh, f_coh, confC] = compute_coherence(Z_true_trials, Z_recon_trials, fs)
% Compute Chronux coherence for each latent dimension across trials
% Inputs:
%   Z_true_trials: [T_trial x N_trials x F]
%   Z_recon_trials: [T_trial x N_trials x F]
%   fs: sampling frequency
% Outputs:
%   coh: [nFreq x F] coherence values
%   f_coh: frequency vector
%   confC: [nFreq x F] confidence intervals

    params.Fs = fs;
    params.tapers = [3 5];
    params.pad = 0;
    params.trialave = 1;
    params.err = [2 0];
    
    [T_trial, N_trials, F] = size(Z_true_trials);
    
    % Use one to infer frequency dimension
    [~, ~, ~, ~, ~, f_coh] = coherencyc(Z_true_trials(:,:,1), Z_recon_trials(:,:,1), params);
    nFreq = length(f_coh);
    
    coh = zeros(nFreq, F);
    confC = zeros(nFreq, F);
    
    for f = 1:F
        [coh(:,f), ~, ~, ~, ~, ~, confC(:,f),~,~] = coherencyc(Z_true_trials(:,:,f), Z_recon_trials(:,:,f), params);
    end
end