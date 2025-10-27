function [s_eeg_like, h_f_processed] = spike_to_eeg(s_i, h_f, param, group_size, smooth_sigma)
% Converts spike matrix into EEG-like signal using alpha kernel convolution
%
% Inputs:
%   s_i        : Neuron × Time binary spike matrix
%   h_f        : Latent field matrix (Time × N_F)
%   param      : Struct with .dt (time step)
%   tau        : time constant for alpha kernel (in bins)
%   group_size : number of neurons per EEG-like channel (e.g., 8)
%
% Outputs:
%   s_eeg_like    : EEG-like matrix (Channels × Time)
%   h_f_processed : Latent variables aligned with EEG length (Time × N_F)

% ------------------------
% 1. Preprocess Inputs
% ------------------------
s_i_double = double(s_i);

% Skip the first half-second transient
half_a_sec = round(0.5 / param.dt);
s_i_pt = s_i_double(:, half_a_sec:end);
h_f_pt = h_f(half_a_sec:end, :);

[N, T_new] = size(s_i_pt);

% ------------------------
% 2. Add inhibitory neurons
% ------------------------
if ~isfield(param, 'inhib_frac')
    param.inhib_frac = 0.2;  % default 20% inhibitory
end
nInhib = round(param.inhib_frac * N);
inhib_idx = randperm(N, nInhib);
s_i_pt_inhib = s_i_pt;
s_i_pt_inhib(inhib_idx, :) = -s_i_pt(inhib_idx, :);  % inhibitory neurons as negative

% ------------------------
% 3. Biphasic Alpha Kernel (with hyperpolarization)
% ------------------------
% Alpha kernel: rise-then-decay shape
tau_pos = 0.002;       % alpha time constant for positive lobe (seconds), e.g. 10 ms
tau_neg = 0.03;       % time constant for negative lobe (if using biphasic phys model)
tmax_factor = 6;      % how many taus to include (8 is safe)
dt_kernel = param.dt; % kernel sampling step (use same as sim or coarser)

% --- Build common time vector (seconds) ---
tmax = tmax_factor * max(tau_pos, tau_neg);
t = 0:dt_kernel:tmax;  % same t for all kernels

% --- Alpha kernel (canonical) ---
alpha_k = (t ./ tau_pos) .* exp(-t ./ tau_pos);
alpha_k = alpha_k / sum(alpha_k);   % normalize by area (or use max)

% --- Biphasic kernel (two-EXP difference approach, phys-like) ---
% positive lobe (fast) and negative lobe (slower, scaled)
biphasic_k = (t ./ tau_pos).*exp(-t./tau_pos) - 0.4*(t ./ tau_neg).*exp(-t./tau_neg);
% center/sink to zero mean if you want
biphasic_k = biphasic_k - mean(biphasic_k);
% normalize by peak absolute value (optional, to compare shapes)
biphasic_k = biphasic_k / max(abs(biphasic_k));

% ------------------------
% 4. Group Neurons + Convolution
% ------------------------
num_channels = floor(N / group_size);
s_convolved = zeros(num_channels, T_new);

for ch = 1:num_channels
    idx = (ch - 1)*group_size + (1:group_size);
    conv_sum = zeros(1, T_new);
    for i = idx
        conv_sum = conv_sum + conv(s_i_pt_inhib(i, :), alpha_k, 'same');
    end
    s_convolved(ch, :) = conv_sum;
end

% ------------------------
% 4.5 Gaussian Smoothing
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
% 5. Z-score Normalization
% ------------------------
s_eeg_like = zscore(s_smoothed, 0, 2);

% ------------------------
% 5b. Convolve True Latent Variable with Biphasic Alpha Kernel
% ------------------------
% Convolve each latent factor with the same biphasic kernel
h_f_conv = zeros(size(h_f_pt));

for f = 1:size(h_f_pt, 2)
    h_f_conv(:, f) = conv(h_f_pt(:, f), alpha_k, 'same');
end

% Optionally normalize or z-score for comparability
h_f_conv = zscore(h_f_conv, 0, 1);

% ------------------------
% 6. Align Latent Variables
% ------------------------
h_f_processed = h_f_conv;
% h_f_processed = h_f_pt;
% Truncate if lengths mismatch
if size(s_eeg_like, 2) ~= size(h_f_processed, 1)
    warning('Truncating EEG signal to match latent variable length.');
    min_len = min(size(s_eeg_like, 2), size(h_f_processed, 1));
    s_eeg_like   = s_eeg_like(:, 1:min_len);
    h_f_processed = h_f_processed(1:min_len, :);
end

end
