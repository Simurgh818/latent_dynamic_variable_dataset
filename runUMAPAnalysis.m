function [outUMAP] = runUMAPAnalysis( ...
        n_neighbors, min_dist, s_train, s_test, param, ...
        h_train, h_test, num_sig_components, results_dir)
% runUMAPAnalysis UMAP reduction + Linear Mapping + Detailed Plotting
%
% Inputs:
%   s_train, s_test     : (Neurons x Time)
%   h_train, h_test     : (Time x Latents)
%   num_sig_components  : dimensionality of UMAP (n_components)
%   results_dir         : Output directory

%% 2. Setup and Directory
method_name = 'UMAP';
method_dir = fullfile(results_dir, method_name);
if ~exist(method_dir, 'dir')
    mkdir(method_dir);
end
file_suffix = sprintf('_n%d_dist%.1f_k%d', n_neighbors, min_dist, num_sig_components);
h_f_colors = lines(param.N_F);

%% 3. Prepare Data (Transpose to Time x Neurons)
% changing the shape to promote learning across time
eeg_train = double([s_train(:, 1:end-1) ; s_train(:, 2:end)])';
eeg_test = double([s_test(:, 1:end-1) ; s_test(:, 2:end)])';

% fix h_test and h_train size 
h_train = h_train(1:end-1,:);
h_test = h_test(1:end-1,:);
rng(42,'twister');

%% 4. Run UMAP (Train) & Transform (Test)
disp(['Running UMAP (k=' num2str(num_sig_components) ') on Training Set...']);

umap_calc_components = max(2, num_sig_components);

% 1. Compute PCA for initialization
[~, score] = pca(eeg_train); 
init_coords = score(:, 1:umap_calc_components);
init_coords = 10 * (init_coords / max(abs(init_coords(:))));

% Run UMAP on training data (We no longer need 'save_template_file')
try
    [umap_train_raw, ~] = run_umap(eeg_train, ...
        'n_neighbors', n_neighbors, ...
        'min_dist', min_dist, ...
        'n_components', umap_calc_components,...        
        'method', 'MEX', ...              
        'verbose', 'none', ...
        'metric', 'euclidean', ...
        'init', init_coords);    
catch ME
    warning('UMAP Train failed: %s', ME.message);
    umap_train_raw = nan(size(eeg_train, 1), umap_calc_components);
end 

disp('Manually Projecting Test Set via KNN Interpolation...');
% =========================================================================
% MANUAL UMAP PROJECTION
% We approximate the UMAP transform by finding the nearest neighbors in the 
% high-dimensional EEG space, and averaging their coordinates in the UMAP space.
% =========================================================================

% We use the same number of neighbors for interpolation
k_interp = max(3, n_neighbors); 

try
    % 1. Find the K nearest training points for every test point
    [idx, D] = knnsearch(eeg_train, eeg_test, 'K', k_interp);
    
    umap_test_raw = zeros(size(eeg_test, 1), umap_calc_components);
    
    % 2. Calculate the weighted average of their UMAP coordinates
    for i = 1:size(eeg_test, 1)
        dists = D(i, :);
        
        % Convert distances to weights (closer = higher weight)
        % Add small epsilon (1e-6) to prevent division by zero if points are identical
        weights = 1 ./ (dists + 1e-6); 
        weights = weights / sum(weights); % Normalize to sum to 1
        
        % Get the UMAP coordinates of these specific training points
        neighbor_coords = umap_train_raw(idx(i, :), :);
        
        % The test point's embedding is the weighted average
        umap_test_raw(i, :) = weights * neighbor_coords;
    end
catch ME
    warning('Manual KNN Projection failed: %s', ME.message);
    umap_test_raw = nan(size(eeg_test, 1), umap_calc_components);
end

% Extract only the requested number of components for analysis
umap_train = umap_train_raw(:, 1:num_sig_components);
umap_test  = umap_test_raw(:, 1:num_sig_components);

%% 6. Compute Performance Metrics
[h_rec_train_final, h_rec_test_final, Comp_latent_matching_corr, R_UMAP, direct_Component_Corr_umap, UMAP_R2_scores, freq_data] = ...
    computePerformanceMetrics(umap_train, umap_test, h_train, h_test, 'UMAP', num_sig_components, param);

%% ============================================================
% PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if (isempty(getCurrentTask()) & num_sig_components==6)    
    % Plot 1: UMAP Embedding (Training) colored by Latents
    cluster_idx = kmeans(h_test, param.N_F,'MaxIter', 1000, 'Replicates', 5, 'Display', 'off');
    
    fig1 = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');
    plot_comps = min(size(umap_test_raw,2), max(2, num_sig_components)); 
    t = tiledlayout(ceil((plot_comps-1)/3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for d = 2:plot_comps
        nexttile;
        gscatter(umap_test_raw(:,1), umap_test_raw(:,d), cluster_idx, [],[],10);
        xlabel('UMAP Dim 1'); ylabel(['UMAP Dim ' num2str(d)]);
        title(['Dim 1 vs. ' num2str(d)]); grid on; legend off;
    end
    title(t, {'UMAP (Train) Colored by Latent Clusters', ...
              ['n=' num2str(n_neighbors) ', dist=' num2str(min_dist) ', (k=' num2str(num_sig_components) ')']}, ...
              'FontSize', 14, 'FontWeight', 'bold');
    colormap(turbo);
    set(findall(fig1,'-property','FontSize'),'FontSize',26);
    saveas(fig1, fullfile(method_dir, ['UMAP_Embedding' file_suffix '.png']));
    close(fig1);
    
    %% Plot 1.1: UMAP Dim 1 vs. Dim 2 colored by Intensity 
    n_latents = size(h_test, 2);
    fig11 = figure('Position', [100, 100, 1600, 900], 'Visible', 'off');
    t = tiledlayout(ceil(n_latents/3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for i = 1:n_latents
        nexttile;
        scatter(umap_test_raw(:,1), umap_test_raw(:,2), 15, h_test(:,i), 'filled');
        xlim([-10 10]); xticks(-10:10:10);
        
        rho = round(direct_Component_Corr_umap(i), 2);
        peak_lbl = num2str(param.f_peak(i));
        title([sprintf('Z_{%s}', peak_lbl), ', \rho = ', num2str(rho)]); 
        
        xlabel('UMAP 1'); ylabel('UMAP 2');
        colormap(turbo); colorbar; clim([-4 4]); 
        axis square; grid on;
    end
    title(t, 'UMAP colored by Intensity of each Latent Variable','FontSize',30);
    set(findall(fig11,'-property','FontSize'),'FontSize',28);
    saveas(fig11, fullfile(method_dir, ['UMAP_Embedding_perLatentVariable' file_suffix '.png']));
    close(fig11);
    
    %% Plot 2: Time Domain Reconstruction (Test Set)
    plotTimeDomainReconstruction(h_test, h_rec_test_final, param, 'UMAP', num_sig_components, direct_Component_Corr_umap, method_dir);
    
    % Embedding Traces:
    plotCTraces(num_sig_components, param, umap_test_raw, method_dir, file_suffix);
    
    %% Plot 4: Band Power Bar Chart & FFT     
    save_path_fft = fullfile(method_dir, ['UMAP_FFT_True_vs_Recon' file_suffix '.png']);
    plotFrequencySpectra(freq_data.Ht_avg, freq_data.Hr_avg, freq_data.f_plot, 'UMAP', param, num_sig_components, save_path_fft);
    
    br2_path = fullfile(method_dir, ['UMAP_Bandwise_R2' file_suffix '.png']);
    plotBandwiseR2(freq_data.R2_avg, freq_data.f_axis, param, num_sig_components, 'UMAP', br2_path);
    
    %% Plot 5: Coherence Analysis (Chronux)
    params_coh.Fs = param.fs; 
    params_coh.tapers = [3 5]; 
    params_coh.pad = 0;
    params_coh.err = [0 0]; 
    movingwin = [1 0.05]; 
    
    fig4 = figure('Position',[50 50 1000 600], 'Visible', 'off');
    tiledlayout(2, ceil(param.N_F/2), 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['UMAP Coherence Analysis (Test Set, k=' num2str(num_sig_components) ')']);
    
    for i = 1:param.N_F
        nexttile;
        try
            [C,~,~,~,~,t_coh,f_coh] = cohgramc(h_test(:, i), h_rec_test_final(:, i), movingwin, params_coh);
            imagesc(t_coh, f_coh, C'); axis xy;
            xlabel('Time (s)'); ylabel('Freq (Hz)');
            ylim([0 50]); clim([0 1]); colorbar;
        catch
            text(0.5,0.5,'Chronux Toolbox missing or error','HorizontalAlignment','center');
        end
        title(['Latent ' num2str(i)]);
    end
    set(findall(fig4,'-property','FontSize'),'FontSize',20);
    saveas(fig4, fullfile(method_dir, ['UMAP_Coherence' file_suffix '.png']));
    close(fig4);
    
    %% Plot 6: Scatter Mean Band Amplitudes
    nBands = numel(freq_data.band_names);
    fig5 = figure('Position',[50 50 1400 300], 'Visible', 'off');
    tiledlayout(1, nBands, 'TileSpacing', 'loose', 'Padding', 'compact');
    sgtitle(['UMAP Band Mean FFT Amplitudes (k=' num2str(num_sig_components) ')']);
    colors = lines(nBands); markers = {'o','s','d','h','^'};
    
    for b = 1:nBands    
        nexttile; hold on;
        x = mean(freq_data.true_vals{b}, 2, 'omitnan');
        y = mean(freq_data.recon_vals{b}, 2, 'omitnan');
        std_x = std(freq_data.true_vals{b}, 0, 2, 'omitnan');
        std_y = std(freq_data.recon_vals{b}, 0, 2, 'omitnan');
        
        for m = 1:length(markers)
            if m>numel(x), break; end
            scatter(x(m), y(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
            errorbar(x(m), y(m), std_x(m), std_y(m), 'LineStyle', 'none', 'Color', colors(b,:));
        end
        plot([min(x) max(x)], [min(x) max(x)], 'Color', colors(b,:), 'LineWidth', 2);
        
        R = corrcoef(x,y); 
        if numel(R) > 1, r_sq = R(1,2)^2; else, r_sq = 0; end
        text(mean(x), mean(y), sprintf('R^2=%.2f', r_sq), 'Color', colors(b,:));
        
        title(freq_data.band_names{b}); grid on;
    end
    set(findall(fig5,'-property','FontSize'),'FontSize',20);
    saveas(fig5, fullfile(method_dir, ['UMAP_Scatter_Mean' file_suffix '.png']));
    close(fig5);
    
    %% Plot 7: Scatter Per-Trial
    plotBandScatterPerTrial(freq_data.true_vals, freq_data.recon_vals, UMAP_R2_scores, freq_data.band_names, param, num_sig_components, "UMAP", method_dir);
end

%% Output Structure 
outUMAP = struct();
outUMAP.umap_train = umap_train;
outUMAP.h_recon_test = h_rec_test_final;
outUMAP.h_recon_train = h_rec_train_final;
outUMAP.n_neighbors = n_neighbors;
outUMAP.min_dist = min_dist;
outUMAP.Comp_latent_matching_corr = Comp_latent_matching_corr;
outUMAP.Comp_latent_matching_matrix = R_UMAP; 
outUMAP.direct_Component_Corr = direct_Component_Corr_umap;
outUMAP.spectral_R2 = UMAP_R2_scores; 
close All;

end