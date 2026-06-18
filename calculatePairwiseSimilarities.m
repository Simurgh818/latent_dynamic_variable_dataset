function pair_metrics = calculatePairwiseSimilarities(W_true, perf_scores)
    if size(W_true, 2) > 100
        error('CRITICAL: W_true has %d columns. This is a time-series matrix!', size(W_true, 2));
    end
    
    nLatents = size(W_true, 2);
    W_norm = W_true ./ vecnorm(W_true, 2, 1);
    CosSimMatrix = abs(W_norm' * W_norm);
    
    pairs = nchoosek(1:nLatents, 2);
    num_pairs = size(pairs, 1);
    
    pair_overlap = zeros(num_pairs, 1);
    pair_perf_diff = zeros(num_pairs, 1);
    pair_perf_mean = zeros(num_pairs, 1); 
    
    % --- NEW: Track indices ---
    pair_idx_A = zeros(num_pairs, 1);
    pair_idx_B = zeros(num_pairs, 1);
    
    for p = 1:num_pairs
        idx_A = pairs(p, 1);
        idx_B = pairs(p, 2);
        
        pair_idx_A(p) = idx_A;
        pair_idx_B(p) = idx_B;
        
        pair_overlap(p) = CosSimMatrix(idx_A, idx_B);
        pair_perf_diff(p) = abs(perf_scores(idx_A) - perf_scores(idx_B));
        pair_perf_mean(p) = (perf_scores(idx_A) + perf_scores(idx_B)) / 2; 
    end
    
    pair_metrics.overlap = pair_overlap;
    pair_metrics.perf_diff = pair_perf_diff;
    pair_metrics.perf_mean = pair_perf_mean; 
    
    % --- Output indices ---
    pair_metrics.idx_A = pair_idx_A;
    pair_metrics.idx_B = pair_idx_B;
end