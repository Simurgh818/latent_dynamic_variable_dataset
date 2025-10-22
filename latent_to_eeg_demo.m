% latent_to_eeg_demo.m
% Generates 32-channel EEG-like data from 3 latent variables using a frozen MLP.
% Usage:
%   [eeg, t, params] = latent_to_eeg_demo;           % runs demo with default settings
%   [eeg, t] = latent_to_eeg(z, Fs);                % use your own z (3 x T)
%   [eeg, t] = latent_to_eeg(z, Fs, params);        % override parameters

function [eeg, t, params] = latent_to_eeg_demo()
    % Demo wrapper: generate default latent variables and run MLP -> EEG.
    Fs = 512;                 % sampling rate Hz
    dur = 10;                 % seconds
    T = dur * Fs;
    z = make_default_z(T, Fs);  % 3 x T
    [eeg, t, params] = latent_to_eeg(z, Fs);
    
    % quick plot
    figure('Position',[100 100 1000 900]);
    subplot(3,1,1);
    plot(t, z');
    title('Latent variables (3)');
    legend('beta-latent (transient)','alpha-latent (continuous)','gamma-latent (occasional)');
    xlabel('Time (s)');

    subplot(3,1,2);
    num_channels_plot = 4;
    offset = 10 * std(eeg(1,:), 1);  % vertical separation between traces
    hold on
    for ch = 1:num_channels_plot
        plot(t*1000, eeg(ch,:) + (num_channels_plot - ch)*offset, 'b');
    end
    hold off
    xlim([0, max(t)*1000]);
    ylim([-offset, num_channels_plot*offset]);
    xlabel('Time (ms)');
    ylabel('Channels (stacked)');
    yticks((0:num_channels_plot-1)*offset);
    yticklabels(arrayfun(@(c) sprintf('Ch %d', num_channels_plot - c + 1), 1:num_channels_plot, 'UniformOutput', false));
    title('EEG-like stacked channels - biphasic alpha kernel');

    subplot(3,1,3);
    imagesc(t, 1:4, eeg);
    axis xy;
    title('EEG (32 channels)');
    xlabel('Time (s)');
    ylabel('Channel');
end

function [eeg_out, t, params] = latent_to_eeg(z, Fs, params)
% latent_to_eeg - maps latent variables z (3 x T) through frozen MLP to 32-channel EEG-like signal
%
% Inputs:
%   z      : 3 x T latent timecourses (if empty, the function will create defaults)
%   Fs     : sampling rate in Hz
%   params : (optional) struct with fields to override defaults
%
% Outputs:
%   eeg_out: 32 x T EEG-like signals
%   t      : time vector (1 x T)
%   params : used params (struct)
%
    if nargin < 2 || isempty(Fs), Fs = 250; end
    if nargin < 3, params = struct(); end

    rng_seed = get_or_default(params, 'rng_seed', 42);
    rng(rng_seed);  % reproducible frozen weights / mixing

    % defaults
    nLatent = 3;
    nChannels = get_or_default(params, 'nChannels', 32);
    hidden_sizes = get_or_default(params, 'hidden_sizes', [64, 128, 64]); % 3 hidden layers
    out_size = nChannels;
    activation_hidden = get_or_default(params, 'activation_hidden', 'tanh'); % 'tanh' or 'relu'
    noise_std = get_or_default(params, 'noise_std', 1e-2);   % sensor noise STD
    apply_spatial_mixing = get_or_default(params, 'apply_spatial_mixing', true);

    % time
    [nLatent_check, T] = size(z);
    if nargin < 1 || isempty(z)
        z = make_default_z(10*Fs, Fs); % fallback 10 seconds if no z provided
        [nLatent_check, T] = size(z);
    end
    if nLatent_check ~= nLatent
        error('z must be of size 3 x T (3 latent variables).');
    end
    t = (0:T-1)/Fs;

    % Build frozen MLP weights (we will not train/update them)
    layer_sizes = [nLatent, hidden_sizes, out_size];
    L = length(layer_sizes) - 1; % number of weight layers

    % Initialize weights and biases (frozen)
    W = cell(L,1);
    b = cell(L,1);
    for li = 1:L
        fan_in = layer_sizes(li);
        fan_out = layer_sizes(li+1);
        % Glorot uniform initialization
        lim = sqrt(6 / (fan_in + fan_out));
        W{li} = -lim + (2*lim).*rand(fan_out, fan_in);
        b{li} = zeros(fan_out,1);
        % small scaling to keep outputs reasonable
        W{li} = W{li} * get_or_default(params, 'weight_scale', 0.8);
    end

    % Forward pass (vectorized over time)
    A = z;  % input: nLatent x T
    for li = 1:L-1  % hidden layers
        Z_linear = W{li} * A + b{li}*ones(1, T);   % fan_out x T
        A = activation(Z_linear, activation_hidden);
    end
    % Output linear layer (before output activation)
    Y_linear = W{L} * A + b{L}*ones(1, T);  % out_size x T

    % Spatial mixing (optional) - this gives correlated sensor topographies. If
    % you want per-channel independent outputs, set apply_spatial_mixing=false.
    if apply_spatial_mixing
        % mixing matrix M (nChannels x out_size) -- for diversity
        M = random_orthonormal_matrix(nChannels, out_size) * (0.8 + 0.2*randn(nChannels, out_size));
        Y_mixed = M * Y_linear;  % nChannels x T
    else
        % If out_size == nChannels then just use Y_linear, otherwise project
        if size(Y_linear,1) == nChannels
            Y_mixed = Y_linear;
        else
            M = random_orthonormal_matrix(nChannels, size(Y_linear,1));
            Y_mixed = M * Y_linear;
        end
    end

    % Output activation: temporal biphasic alpha-like kernel (convolution across time)
    kernel_dur = get_or_default(params, 'kernel_dur', 0.35); % seconds
    kernel = biphasic_alpha_kernel(Fs, kernel_dur);
    % Convolve each channel with kernel (same kernel for every channel)
    eeg_conv = zeros(size(Y_mixed));
    for ch = 1:size(Y_mixed,1)
        eeg_conv(ch,:) = conv(Y_mixed(ch,:), kernel, 'same');
    end

    % Optionally apply a pointwise nonlinearity (e.g., tanh saturating) -
    % keep mild to preserve linear mixing
    eeg_nl = tanh(1.2 * eeg_conv);

    % Add sensor noise
    eeg_out = eeg_nl + noise_std * randn(size(eeg_nl));

    % return used params
    params.W = W; params.b = b; params.M = get_or_default(params,'M',[]);
    params.kernel = kernel;
    params.t = t;
    params.Fs = Fs;
end

%% Helper utilities -------------------------------------------------------
function val = get_or_default(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end

function A = activation(X, name)
    switch lower(name)
        case 'tanh'
            A = tanh(X);
        case 'relu'
            A = max(0, X);
        case 'leakyrelu'
            A = max(0.01*X, X);
        otherwise
            A = tanh(X);
    end
end

function K = biphasic_alpha_kernel(Fs, dur)
% biphasic_alpha_kernel - returns a biphasic alpha-like kernel (zero-mean) lasting ~dur seconds
% We construct it as the difference of two gamma-like impulse responses
    t = (0:1/Fs:dur)';
    % parameters tuned to produce a small positive lobe followed by negative undershoot
    tau1 = 0.06;  amp1 = 1.0;
    tau2 = 0.12;  amp2 = 0.7;
    g1 = (t./tau1).*exp(1 - t./tau1);   % simple alpha-like gamma
    g2 = (t./tau2).*exp(1 - t./tau2);
    K_raw = amp1 * g1 - amp2 * g2;
    % bandlimit / smooth a bit using small Gaussian (or normalize)
    K = K_raw - mean(K_raw);  % zero-mean
    K = K / max(abs(K));      % normalize amplitude to 1
end

function M = random_orthonormal_matrix(rows, cols)
% returns a rows x cols matrix with near-orthonormal columns (via QR)
    A = randn(rows, cols);
    if rows >= cols
        [Q,~] = qr(A,0);
        M = Q(:,1:cols);
    else
        % fewer rows than cols: orthonormal rows via QR of A'
        [Q,~] = qr(A',0);
        M = Q(1:rows, :)';
    end
end

function z = make_default_z(T, Fs)
% Create 3 latent variables with requested properties:
% 1) beta bursts (12-30 Hz) transient every 3.5 s for 0.5 s
% 2) continuous alpha (8-12 Hz) - sustained sinusoid with some jitter
% 3) occasional gamma (30-50 Hz) bursts (short)
    t = (0:T-1)/Fs;
    z = zeros(3, T);

    % 1) beta-latent: bursts every 3.5s lasting 0.5s (center frequency 20Hz)
    burst_interval = 3.5; burst_dur = 0.5;
    burst_centers = 0:burst_interval:max(t);
    beta_freq = 20;  % center
    for c = burst_centers
        start_idx = round((c)/1 * Fs) + 1;
        win_idx = start_idx : min(start_idx + round(burst_dur*Fs)-1, T);
        if isempty(win_idx), continue; end
        amp_env = tukeywin(length(win_idx), 0.5)'; % smooth ramp
        z(1, win_idx) = z(1, win_idx) + amp_env .* sin(2*pi*beta_freq*(0:length(win_idx)-1)/Fs);
    end
    % small baseline noise / low-level spontaneous
    z(1,:) = z(1,:) + 0.05 * randn(1,T);

    % 2) alpha-latent: continuous 10Hz with slow amplitude modulation
    alpha_freq = 10;
    amp_mod = 0.6 + 0.4 * sin(2*pi*0.1*t + 0.3); % slow mod ~0.1Hz
    z(2,:) = amp_mod .* sin(2*pi*alpha_freq*t) + 0.02*randn(1,T);

    % 3) gamma-latent: occasional short bursts (30-50Hz)
    gamma_freqs = 30 + 20*rand(1,round(max(t)/(5)+1));  % random center per burst
    % schedule bursts randomly with Poisson approx (mean every ~6-8s)
    p_mean = 8; % average inter-burst seconds
    burst_onsets = 0;
    current = 0;
    while current < max(t)
        isi = exprnd(p_mean);
        current = current + isi;
        if current > max(t), break; end
        burst_onsets(end+1) = current; %#ok<AGROW>
    end
    for bi = 2:length(burst_onsets)
        c = burst_onsets(bi);
        f = 35 + 10*rand(); % random center 30-45Hz
        bd = 0.12 + 0.08*rand(); % 120-200ms
        start_idx = round(c*Fs) + 1;
        win_idx = start_idx : min(start_idx + round(bd*Fs)-1, T);
        if isempty(win_idx), continue; end
        amp_env = tukeywin(length(win_idx), 0.7)';
        z(3, win_idx) = z(3, win_idx) + 0.5*amp_env .* sin(2*pi*f*(0:length(win_idx)-1)/Fs);
    end
    % add slight noise
    z(3,:) = z(3,:) + 0.03*randn(1,T);
end
