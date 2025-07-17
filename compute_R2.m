function R2_vals = compute_R2(Z_true, Z_recon)
% Compute R^2 between true and reconstructed latent variables
% Inputs:
%   Z_true: [T x F] true latent variables
%   Z_recon: [T x F] reconstructed latent variables
% Output:
%   R2_vals: [1 x F] vector of R^2 values

    SS_tot = sum((Z_true - mean(Z_true)).^2);
    SS_res = sum((Z_true - Z_recon).^2);
    R2_vals = 1 - SS_res ./ SS_tot;
end