function pair_metrics = calculatePairwiseSimilarities(W_true, perf_scores)
    % W_true: The spatial mixing matrix of size [nChannels x nLatents]
    % perf_scores: The recovery correlation for each latent [nLatents x 1]
    
    % --- SAFETY GUARD AGAINST TIME-SERIES MATRICES ---
    if size(W_true, 2) > 100
        error(['CRITICAL: W_true has %d columns. This is a time-series ' ...
               'matrix, not a spatial mixing matrix! Stop before crashing RAM.'], size(W_true, 2));
    end
    
    nLatents = size(W_true, 2);
    
    % Normalize column vectors and get Cosine Similarity Matrix
    W_norm = W_true ./ vecnorm(W_true, 2, 1);
    CosSimMatrix = abs(W_norm' * W_norm);
    
    % Get all unique pairs of latents
    pairs = nchoosek(1:nLatents, 2);
    num_pairs = size(pairs, 1);
    
    % Preallocate output arrays
    pair_overlap = zeros(num_pairs, 1);
    pair_perf_diff = zeros(num_pairs, 1);
    pair_perf_mean = zeros(num_pairs, 1);

    % Calculate metrics for each pair
    for p = 1:num_pairs
        idx_A = pairs(p, 1);
        idx_B = pairs(p, 2);
        
        pair_overlap(p) = CosSimMatrix(idx_A, idx_B);
        
        % Absolute difference in performance
        pair_perf_diff(p) = abs(perf_scores(idx_A) - perf_scores(idx_B));
        
        % Mean performance of the pair
        pair_perf_mean(p) = (perf_scores(idx_A) + perf_scores(idx_B)) / 2;
    end
    
    % Save to output structure
    pair_metrics.overlap = pair_overlap;
    pair_metrics.perf_diff = pair_perf_diff;
    pair_metrics.perf_mean = pair_perf_mean;

end