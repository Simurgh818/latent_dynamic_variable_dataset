function [s_eeg_like, h_f_processed] = spike_to_eeg(s_i, h_f, param, tau, group_size)
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
s_i_pt(inhib_idx, :) = -s_i_pt(inhib_idx, :);  % inhibitory neurons as negative

% ------------------------
% 3. Biphasic Alpha Kernel (with hyperpolarization)
% ------------------------
t_kernel = 0:round(6 * tau);
alpha_kernel = (t_kernel / tau) .* exp(-t_kernel / tau);
alpha_kernel = alpha_kernel / sum(alpha_kernel);

% Add small negative tail to mimic hyperpolarization
neg_tail_shift = round(0.1 * length(alpha_kernel));  % small delay for negative lobe
biphasic_kernel = alpha_kernel - 0.3 * circshift(alpha_kernel, neg_tail_shift);
biphasic_kernel = biphasic_kernel - mean(biphasic_kernel);  % center around zero

% ------------------------
% 4. Group Neurons + Convolution
% ------------------------
num_channels = floor(N / group_size);
s_convolved = zeros(num_channels, T_new);

for ch = 1:num_channels
    idx = (ch - 1)*group_size + (1:group_size);
    conv_sum = zeros(1, T_new);
    for i = idx
        conv_sum = conv_sum + conv(s_i_pt(i, :), biphasic_kernel, 'same');
    end
    s_convolved(ch, :) = conv_sum;
end

% ------------------------
% 5. Z-score Normalization
% ------------------------
s_eeg_like = zscore(s_convolved, 0, 2);

% ------------------------
% 6. Align Latent Variables
% ------------------------
h_f_processed = h_f_pt;

% Truncate if lengths mismatch
if size(s_eeg_like, 2) ~= size(h_f_processed, 1)
    warning('Truncating EEG signal to match latent variable length.');
    min_len = min(size(s_eeg_like, 2), size(h_f_processed, 1));
    s_eeg_like   = s_eeg_like(:, 1:min_len);
    h_f_processed = h_f_processed(1:min_len, :);
end

end
