function [corr_table, R] = match_components_to_latents(C, H, method_name)
    % C: T x K   (components)
    % H: T x N_F (true latents)

    % --- z-score in time ---
    Cz = zscore(C);
    Hz = zscore(H);

    % --- correlation: latents x components ---
    R = corr(Hz, Cz);   % size: N_F x K

    % --- best match per latent ---
    [best_corr, comp_idx] = max(abs(R), [], 2);

    % --- build table ---
    corr_table = table( ...
        (1:size(Hz,2))', ...
        comp_idx, ...
        best_corr, ...
        'VariableNames', {'h_f', 'component_idx', 'corr_value'} ...
    );

    corr_table.method = repmat(string(method_name), height(corr_table), 1);
end
