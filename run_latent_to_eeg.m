function [eeg_out, h_f_processed] = run_latent_to_eeg(Z, Fs, nChannels)
% run_latent_to_eeg  Map latent variables (3 x T) -> EEG-like signals (nChannels x T)
% - Simple frozen MLP with explicit weight matrices
% - Convolution with biphasic alpha-like kernel as output activation
% - Returns nChannels x T matrix

    rng(42); % reproducible frozen weights

    [nLatent, T] = size(Z);
    if nLatent ~= 3
        error('Z must be 3 x T (3 latents).');
    end

    %% MLP architecture
    hidden_sizes = [128, 64, 64]; % you can change to 2-3 hidden layers
    layer_sizes = [nLatent, hidden_sizes, nChannels];
    L = numel(layer_sizes)-1;

    % Initialize frozen weights and biases (Glorot uniform-ish)
    W = cell(L,1); b = cell(L,1);
    for li = 1:L
        fan_in = layer_sizes(li);
        fan_out = layer_sizes(li+1);
        lim = sqrt(6/(fan_in + fan_out));
        W{li} = ( -lim + (2*lim).*rand(fan_out, fan_in) ) * 0.8; % scale
        b{li} = zeros(fan_out,1);
    end

    % Forward pass (vectorized)
    A = Z; % size: features x T
    for li = 1:(L-1)
        Zlin = W{li} * A + b{li} * ones(1, T); % fan_out x T
        A = tanh(Zlin); % hidden activation
    end
    Ylin = W{L} * A + b{L} * ones(1, T); % nChannels x T (pre-output)

    % Spatial mixing (to create channel topographies)
    M = orthonormal_mix(nChannels, size(Ylin,1));
    Ymixed = M * Ylin; % nChannels x T

    % Ymixed = Ylin; % nChannels x T
    % Output activation: convolve with biphasic alpha-like kernel
    % kernel = biphasic_alpha_kernel(Fs, 0.35); % 0.35 s kernel
    kernel = alpha_kernel(Fs, 0.35); % 0.35 s kernel
    eeg_conv = zeros(size(Ymixed));
    for ch = 1:size(Ymixed,1)
        eeg_conv(ch,:) = conv(Ymixed(ch,:), kernel, 'same');
    end

    % mild pointwise nonlinearity and add small sensor noise
    eeg_nl = tanh(1.1 * eeg_conv);
    eeg_out = eeg_nl + 5e-3 * randn(size(eeg_nl));
    
    h_f_conv = zeros(size(Z));

    for f = 1:size(Z, 1)
        h_f_conv(f,:) = conv(Z(f,:), kernel, 'same');
    end
    
    % Optionally normalize or z-score for comparability
    h_f_conv = zscore(h_f_conv, 0, 2);
    h_f_processed = h_f_conv;

end


% --- small helpers ----------------------------------------------------
function M = orthonormal_mix(rows, cols)
    % returns rows x cols mixing matrix with reasonable scaling
    A = randn(rows, cols);
    if rows >= cols
        [Q,~] = qr(A,0);
        M = Q(:,1:cols);
    else
        [Q,~] = qr(A',0);
        M = Q(1:rows, :)';
    end
    M = M .* (0.7 + 0.4*randn(size(M))); % add small amplitude variability
end

function K = biphasic_alpha_kernel(Fs, dur)
    % produce a biphasic alpha-like kernel (zero-mean, normalized)
    t = (0:1/Fs:dur)';
    tau1 = 0.06; amp1 = 1.0;
    tau2 = 0.12; amp2 = 0.8;
    g1 = (t./tau1) .* exp(1 - t./tau1);
    g2 = (t./tau2) .* exp(1 - t./tau2);
    Kraw = amp1*g1 - amp2*g2;
    K = Kraw - mean(Kraw);
    K = K / max(abs(K) + eps);
end

function K = alpha_kernel(Fs, dur)
    % --- Alpha kernel (canonical) ---
    t = (0:1/Fs:dur)';
    tau = 0.06; 
    K = (t ./ tau) .* exp(-t ./ tau);
    K = K / sum(K);   % normalize by area (or use max)
end