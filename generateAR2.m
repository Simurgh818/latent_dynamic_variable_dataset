function x = generateAR2(f_peak, zeta, fs, duration)
% generateAR2 - Generate an AR(2) process with given resonance frequency and damping.
%
% Inputs:
%   f_peak   - desired peak frequency (Hz)
%   zeta     - damping ratio (0 < zeta < 1)
%   fs       - sampling frequency (Hz)
%   duration - total duration (seconds)
%
% Output:
%   x        - generated AR(2) time series (column vector)

N = round(fs * duration);
r = exp(-zeta * 2*pi*f_peak / fs);   % pole radius
theta = 2*pi*f_peak / fs;            % pole angle

a1 = 2*r*cos(theta);
a2 = -r^2;

% white noise drive
eps = randn(N,1);

x = zeros(N,1);
x(1:2) = eps(1:2);
for n = 3:N
    x(n) = a1*x(n-1) + a2*x(n-2) + eps(n);
end

% normalize to unit RMS
x = x / rms(x);
end
