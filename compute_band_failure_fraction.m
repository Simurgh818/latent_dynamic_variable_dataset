function frac_fail = compute_band_failure_fraction(coh, f, threshold)
% Compute fraction of frequency bands with low coherence
% Inputs:
%   coh: [nFreq x F] coherence values
%   f: [nFreq x 1] frequency vector
%   threshold: scalar threshold for coherence (e.g., 0.2)
% Output:
%   frac_fail: [F x Nbands] fraction of frequencies < threshold in each band

% Define canonical frequency bands
    bands = [0.5 4; 4 8; 8 12; 12 30; 30 80];
    Nbands = size(bands,1);
    F = size(coh,2);
    
    frac_fail = zeros(F, Nbands);
    
    for b = 1:Nbands
        band_mask = f >= bands(b,1) & f <= bands(b,2);
        for f_i = 1:F
            coh_band = coh(band_mask, f_i);
            frac_fail(f_i, b) = mean(coh_band < threshold);
        end
    end
end