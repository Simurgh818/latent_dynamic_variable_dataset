function [Z_true_trials, Z_recon_trials] = make_trial_matrices(Z_true, Z_recon, L)
% Split latent variable trajectories into trials of fixed length
% Inputs:
%   Z_true: [T x F] matrix of true latent variables
%   Z_recon: [T x F] matrix of reconstructed latent variables
%   trial_length: number of timepoints per trial
% Outputs:
%   Z_true_trials: [T_trial x N_trials x F]
%   Z_recon_trials: [T_trial x N_trials x F]

    [T, F] = size(Z_true);
    N_trials = floor(T / L);
    
    Z_true_trials  = zeros(L, N_trials, F);
    Z_recon_trials = zeros(L, N_trials, F);
    
    for tr = 1:N_trials
        idx = (tr-1)*L + (1:L);        % sample indices for this trial
  
        % extract one trial
        Z_true_trials(:,tr,:)  = Z_true(idx, :);      
        Z_recon_trials(:,tr,:) = Z_recon(idx, :);

    end
end