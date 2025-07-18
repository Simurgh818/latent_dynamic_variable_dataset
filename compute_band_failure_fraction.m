function frac_fail = compute_band_failure_fraction(coh, f, threshold)
% Compute fraction of frequencies in each canonical band where
% coherence < threshold.  Threshold can be either:
%  • a scalar (same cutoff for all freqs) or
%  • an [nFreq×1] vector (frequency‐dependent cutoff, e.g. confC)
%
% Inputs:
%   coh:       [nFreq × F]  coherence values
%   f:         [nFreq × 1]  frequency vector
%   threshold: scalar OR [nFreq×1] vector
%
% Output:
%   frac_fail: [F × Nbands] fraction of frequencies in each band
%              whose coh < threshold

    % Define canonical frequency bands
    bands = [0.5 4; 4 8; 8 12; 12 30; 30 80];
    Nbands = size(bands,1);
    [nFreq, F] = size(coh);
    
    % If threshold is scalar, turn into vector
    if isscalar(threshold)
        threshold = threshold * ones(nFreq,1);
    elseif numel(threshold)~=nFreq
        error('Threshold must be scalar or length(f) vector');
    end
    
    frac_fail = zeros(F, Nbands);
    
    for b = 1:Nbands
        band_mask = f >= bands(b,1) & f <= bands(b,2);
        for fi = 1:F
            coh_band      = coh(band_mask, fi);
            thr_band      = threshold(band_mask);
            % count how many of coh_band fall below their own threshold
            frac_fail(fi,b) = mean(coh_band < thr_band);
        end
    end
end
