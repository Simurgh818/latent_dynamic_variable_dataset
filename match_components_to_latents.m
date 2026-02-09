function [corr_table, R] = match_components_to_latents(C, H, method_name, k)
    % C: T x K_total   (components)
    % H: T x N_F       (true latents)
    % method_name: string
    % k: integer       (limit matching to first k columns of C)
    
    % --- Handle optional k ---
    if nargin < 4 || isempty(k)
        k = size(C, 2);
    end
    
    % --- Slice components to restrict search to first k ---
    % Safety check: don't index more columns than exist
    k_eff = min(k, size(C, 2));
    C_subset = C(:, 1:k_eff);
    
    % --- z-score in time ---
    Cz = zscore(C_subset);
    Hz = zscore(H);

    % --- correlation: latents x components ---
    % R matrix will now be [N_F x k_eff]
    R = corr(Hz, Cz);   
    
    % --- best match per latent ---
    % Finds max correlation along the component dimension (dim 2)
    [best_corr, comp_idx] = max(abs(R), [], 2);
    
    % --- build table ---
    corr_table = table( ...
        (1:size(Hz,2))', ...
        comp_idx, ...
        best_corr, ...
        'VariableNames', {'h_f', 'component_idx', 'corr_value'} ...
    );
    
    % Add metadata
    corr_table.method = repmat(string(method_name), height(corr_table), 1);
    
    % Optional: Store which 'k' was used for this calculation
    corr_table.k_limit = repmat(k_eff, height(corr_table), 1);
end
