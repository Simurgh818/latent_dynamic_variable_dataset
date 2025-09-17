function [s_eeg_like, h_f_processed] = spike_to_eeg(s_i, h_f, param, target_bin_size, tau, group_size, smooth_sigma)
% Converts spike matrix into EEG-like signal with optional latent field processing
%
% Inputs:
%   s_i          : Neuron x Time binary spike matrix
%   h_f          : (optional) Latent field matrix (Time x N_F)
%   param        : Struct with .dt (time step)
%   target_bin_size : desired bin size in seconds (e.g., 0.02)
%   tau          : time constant for alpha kernel (in bins)
%   group_size   : number of neurons per EEG-like channel (e.g., 8)
%   smooth_sigma : std dev for Gaussian smoothing (in bins)
%
% Outputs:
%   s_eeg_like     : EEG-like matrix (Channels × Time)
%   h_f_processed  : (optional) binned & convolved latent variables (Time × N_F)

% ------------------------
% 1. Temporal Binning
% ------------------------
original_bin_size = param.dt;
bin_ratio = target_bin_size / original_bin_size;

[N, T] = size(s_i);
T_new = floor(T / bin_ratio);
s_binned = zeros(N, T_new);

for t = 1:T_new
    idx_start = round((t-1)*bin_ratio + 1);
    idx_end   = round(t*bin_ratio);
    s_binned(:, t) = sum(s_i(:, idx_start:idx_end), 2);
end

% ------------------------
% 2. Alpha Kernel Convolution
% ------------------------
t_kernel = 0:round(6 * tau);
alpha_kernel = (t_kernel / tau) .* exp(-t_kernel / tau);  % canonical alpha kernel
alpha_kernel = alpha_kernel / sum(alpha_kernel);          % normalize

num_channels = floor(N / group_size);
s_convolved = zeros(num_channels, T_new);

for ch = 1:num_channels
    idx_start = (ch - 1)*group_size + 1;
    idx_end   = ch * group_size;
    signal = zeros(1, T_new);
    for i = idx_start:idx_end
        conv_spk = conv(s_binned(i, :), alpha_kernel, 'same');
        signal = signal + conv_spk;
    end
    s_convolved(ch, :) = signal;
end

% ------------------------
% 3. Gaussian Smoothing
% ------------------------
kernel_size = round(6 * smooth_sigma);
x = -kernel_size:kernel_size;
gauss_kernel = exp(-x.^2 / (2 * smooth_sigma^2));
gauss_kernel = gauss_kernel / sum(gauss_kernel);

s_smoothed = zeros(size(s_convolved));
for i = 1:num_channels
    s_smoothed(i, :) = conv(s_convolved(i, :), gauss_kernel, 'same');
end

% ------------------------
% 4. Z-score Normalization
% ------------------------
s_eeg_like = zscore(s_smoothed, 0, 2);

% ------------------------
% 5. (Optional) Process Latent Variables h_f
% ------------------------
if nargin >= 2 && ~isempty(h_f)
    [T_hf, N_F] = size(h_f);
    T_hf_new = floor(T_hf / bin_ratio);
    h_f_binned = zeros(T_hf_new, N_F);

    for t = 1:T_hf_new
        idx_start = round((t-1)*bin_ratio + 1);
        idx_end   = round(t*bin_ratio);
        h_f_binned(t, :) = mean(h_f(idx_start:idx_end, :), 1);
    end

    % Convolve each latent field with alpha kernel
    h_f_processed = zeros(T_hf_new, N_F);
    for f = 1:N_F
        h_f_processed(:, f) = conv(h_f_binned(:, f), alpha_kernel, 'same');
    end

    % Match EEG length if needed
    if size(s_eeg_like, 2) ~= size(h_f_processed, 1)
        warning('Truncating EEG signal to match latent variable length.');
        min_len = min(size(s_eeg_like, 2), size(h_f_processed, 1));
        s_eeg_like   = s_eeg_like(:, 1:min_len);
        h_f_processed = h_f_processed(1:min_len, :);
    end
else
    h_f_processed = [];
end

end