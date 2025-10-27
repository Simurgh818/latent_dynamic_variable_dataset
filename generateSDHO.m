function h_F = generateSDHO(f_peak, zeta_val, dt, T)
% uses transfer function (tf) to generate position vs time for a stochastic
% damped harmonic oscillator with damping ratio zeta_val and resonance
% frequency f_peak. 
% clear; clc;

%% Input parameters
% fs      = 1000;          
% T       = 10;            % duration (s)
fs = 1/dt;                 % sampling rate (Hz)
t       = (0:1/fs:T-1/fs)'; 
N       = numel(t);
% 
% f_peak  = 12;            % desired observable peak frequency (Hz) ~ EEG alpha
% zeta_val    = 0.08;          % damping ratio (0<zeta<1, underdamped)

% rng(1);                  % reproducibility
        
%% Map "observable" peak frequency to natural frequency
% For an underdamped oscillator: f_peak = f_d = f0 * sqrt(1 - zeta^2)
% so f0 = f_peak / sqrt(1 - zeta^2)
f0      = f_peak / sqrt(1 - zeta_val^2);        % natural frequency (Hz)
w0      = 2*pi*f0;                           % natural angular freq
wd      = w0*sqrt(1 - zeta_val^2);               % damped angular freq

%% ===== 1) Continuous-time SDHO -> discretized LTI driven by white noise =====
% SDE (deterministic part): x'' + 2*zeta*w0 x' + w0^2 x = u(t)
% Transfer function from u -> x: 1 / (s^2 + 2*zeta*w0 s + w0^2)

num_c = 2*sqrt(zeta_val)*w0^(3/2);%1;
den_c = [1 2*zeta_val*w0 w0^2];
sys_c = tf(num_c, den_c);

% Discretize the continuous system at fs using zero-order hold
sys_d = c2d(ss(sys_c), 1/fs, 'zoh');   % state-space (more numerically stable)

% Drive with white noise (unit variance) and simulate
u = randn(N,1)*sqrt(fs);
x_ct = lsim(sys_d, u, t);

% Normalize to unit RMS (so the comparison is shape-based, not scale-based)
h_F = x_ct ;%/ rms(x_ct);

% generate SDHO latent variables