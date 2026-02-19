function [R2_train_global, MSE_train_global, outUMAP] = runUMAPAnalysis( ...
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

% File naming suffix
file_suffix = sprintf('_n%d_dist%.1f_k%d', n_neighbors, min_dist, num_sig_components);
h_f_colors = lines(param.N_F);

%% 3. Prepare Data (Transpose to Time x Neurons)
% eeg_train = double(s_train)';     
% eeg_test  = double(s_test)'; 

% ToDO: changing the shape to promote learning across time
eeg_train = double([s_train(:, 1:end-1) ; s_train(:, 2:end)])';
eeg_test = double([s_test(:, 1:end-1) ; s_test(:, 2:end)])';
% fix h_test and h_train size 
h_train = h_train(1:end-1,:);
h_test = h_test(1:end-1,:);
rng(42,'twister');
%% 4. Run UMAP (Train) & Transform (Test)
disp(['Running UMAP (k=' num2str(num_sig_components) ') on Training Set...']);

% WORKAROUND 1: The Meehan run_umap library requires n_components >= 2.
% If num_sig_components is 1, we must request 2, then ignore the 2nd dim.
umap_calc_components = max(2, num_sig_components);

% 1. Compute PCA
% 'score' contains the coordinates of your data in PCA space.
% We only need the first 2 dimensions for the initialization.
[~, score] = pca(eeg_train); 
init_coords = score(:, 1:umap_calc_components);
[~, score_test] = pca(eeg_test); 
init_coords_test = score_test(:, 1:umap_calc_components);

% 2. Normalize (Optional but Recommended)
% UMAP usually prefers the initialization to be within a small range (e.g., -10 to 10).
% You can scale it, though most implementations handle this automatically.
init_coords = 10 * (init_coords / max(abs(init_coords(:))));

% Run UMAP on training data and keep the object (struct) to transform test data
% WORKAROUND 2: 'check_duplicates', false prevents the library from asking 
% "run_umap will remove duplicates..." which halts automation.
try
    [umap_train_raw, umap_struct] = run_umap(eeg_train, ...
        'n_neighbors', n_neighbors, ...
        'min_dist', min_dist, ...
        'n_components', umap_calc_components,...        
        'method', 'MEX', ...              % CRITICAL: Explicitly set method
        'verbose', 'none', ...
        'metric', 'euclidean', ...
        'randomize', true, ...
        'init', init_coords, ...
        'gui', false, ...
        'check_duplicates', false, ...   % FIX: Don't ask to remove duplicates (prevents dialog crash)
        'plot_output', 'none', ...       % FIX: Don't try to plot results (prevents Toolbar crash)
        'save_template_file', false);    % Optimization: Don't save .mat template automatically
catch ME
    warning('UMAP Train failed: %s. Attempting fallback settings.', ME.message);
    rethrow(ME); % Rethrow the error for debugging purposes
end 
disp('Projecting Test Set into UMAP space...');

% Transform test data using the learned training manifold
try
    umap_test_raw = run_umap(eeg_test, ...
        'template', umap_struct, ...
        'method', 'MEX', ...              
        'n_components', umap_calc_components,...
        'verbose', 'none', ...
        'metric', 'euclidean', ...
        'randomize', true, ...
        'init', init_coords_test, ...
        'gui', false, ...
        'check_duplicates', false, ...       % FIX ADDED
        'plot_output', 'none');              % FIX ADDED
catch
    warning('Could not project test data via template. Running UMAP independently on Test.');
    umap_test_raw = run_umap(eeg_test, ...
        'n_neighbors', n_neighbors, ...
        'min_dist', min_dist, ...
        'n_components', umap_calc_components, ...
        'method', 'MEX', ...   
        'metric', 'euclidean', ...
        'randomize', true, ...
        'init', init_coords_test, ...
        'gui', false, ...
        'check_duplicates', false, ...       % FIX ADDED
        'plot_output', 'none');              % FIX ADDED
end


% Extract only the requested number of components for analysis
% If k=1, we take column 1. If k>=2, we take columns 1:k.
umap_train = umap_train_raw(:, 1:num_sig_components);
umap_test  = umap_test_raw(:, 1:num_sig_components);

% Mapping components to latents
C = umap_train;   % T x nComp
H = h_train(1:size(C,1), :);

[corr_UMAP, R_UMAP] = match_components_to_latents(C, H, 'UMAP',num_sig_components);

%% 5. Reconstruction Loop (Train Mapping -> Test Eval)
% We calculate metrics for k=1 to num_sig_components
MSE_train_curve = zeros(1, param.N_F);
R2_train_curve  = zeros(1, param.N_F);
MSE_test_curve = zeros(1, param.N_F);
R2_test_curve  = zeros(1, param.N_F);

disp('Calculating Reconstruction Metrics...');
% 1. Learn Map: UMAP_train -> H_train
% Solve W such that: UMAP_train * W = H_train using all current components
% We use lsqlin or simple pinv. lsqlin is safer for regularization.

% W_k = zeros(umap_calc_components,param.N_F);
% for f = 1:param.N_F
%     W_k(:,f) = lsqlin(umap_test_raw, h_test(:,f));
% end
W_k_train = umap_train_raw \ h_train;
W_k = umap_test_raw \ h_test;
% 2. Apply Map to Test: UMAP_test * W -> H_rec_test
h_rec_train_final = umap_train_raw * W_k_train;
h_rec_test_final = umap_test_raw * W_k;

% 3. Calculate Test Metrics
for f = 1:param.N_F
    % MSE
    MSE_test_curve(1,f) = mean((h_test(:,f) - h_rec_test_final(:,f)).^2);
    MSE_train_curve(1,f) = mean((h_train(:,f) - h_rec_train_final(:,f)).^2);
    % R2
    res_var = sum((h_test(:,f) - h_rec_test_final(:,f)).^2);
    tot_var = sum((h_test(:,f) - mean(h_test(:,f))).^2);
    R2_test_curve(1,f) = 1 - (res_var / tot_var);
    res_var_train = sum((h_train(:,f) - h_rec_train_final(:,f)).^2);
    tot_var_train = sum((h_train(:,f) - mean(h_train(:,f))).^2);
    R2_train_curve(1,f) = 1 - (res_var_train / tot_var_train);
end

% Global outputs (vector of size num_sig_components for the main script loop)
% We take the mean across all latent fields for the global metric
MSE_test_global = mean(MSE_test_curve, 2); 
R2_test_global  = mean(R2_test_curve, 2);
MSE_train_global = mean(MSE_train_curve, 2); 
R2_train_global  = mean(R2_train_curve, 2);
% Normalized version for plotting
h_rec_test_norm = h_rec_test_final; 
for f = 1:param.N_F
   h_rec_test_norm(:,f) = h_rec_test_final(:,f) ./ std(h_rec_test_final(:,f)); 
   h_rec_train_norm(:,f) = h_rec_train_final(:,f) ./ std(h_rec_train_final(:,f)); 
end
fs_new = param.fs;
%% ============================================================
% PLOTTING SECTION (Using TEST Data)
% ============================================================
if isempty(getCurrentTask()) && num_sig_components >4

    % Plot 1: UMAP Embedding (Training) colored by Latents
    % We visualize the Training embedding because that's the manifold structure we learned
    cluster_idx = kmeans(h_train, param.N_F,'MaxIter', 1000, 'Replicates', 5, 'Display', 'off');
    
    fig1 = figure('Position', [100, 100, 1200, 800]);
    % If we only have 1 component, we can't do a 2D scatter properly vs another dim.
    % We default to plotting the 2 components calculated (even if only 1 was requested/used).
    plot_comps = min(size(umap_train_raw,2), max(2, num_sig_components)); 
    
    t = tiledlayout(ceil((plot_comps-1)/3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    % Plot Dim 1 vs others
    for d = 2:plot_comps
        nexttile;
        gscatter(umap_train_raw(:,1), umap_train_raw(:,d), cluster_idx, [],[],10);
        xlabel('UMAP Dim 1'); ylabel(['UMAP Dim ' num2str(d)]);
        title(['Dim 1 vs. ' num2str(d)]); grid on; legend off;
    end
    title(t, {'UMAP (Train) Colored by Latent Clusters', ...
              ['n=' num2str(n_neighbors) ', dist=' num2str(min_dist) ', (k=' num2str(num_sig_components) ')']}, ...
              'FontSize', 14, 'FontWeight', 'bold');
    colormap(turbo);
    set(findall(fig1,'-property','FontSize'),'FontSize',20);
    saveas(fig1, fullfile(method_dir, ['UMAP_Embedding' file_suffix '.png']));
    
    %% Plot 1.1: UMAP Dim 1 vs. Dim 2 colored by Intensity of each latent variable
    %  plot UMAP dim 1 vs. dim 2 for each latent variable 
    n_latents = size(h_train, 2);
    fig11 = figure('Position', [100, 100, 1400, 800]);
    
    % Create a layout: one tile for each Latent Variable
    t = tiledlayout(ceil(n_latents/4), 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for i = 1:n_latents
        nexttile;
        % We color by the i-th column of your h_f matrix
        scatter(umap_train_raw(:,1), umap_train_raw(:,2), 15, h_train(:,i), 'filled');
        r2= round(R2_test_curve(end,i),2);
        title(['Latent Variable ' num2str(i), ', R^2= ' num2str(r2)]); % Or use a name like "Freq 12Hz"
        xlabel('UMAP 1'); ylabel('UMAP 2');
        colormap(turbo); 
        colorbar; % Shows the scale of the latent value
        clim([-4 4]); % Set the colorbar range from -4 to 4
        axis square; grid on;
    end
    
    title(t, 'UMAP colored by Intensity of each Latent Variable');
    set(findall(fig11,'-property','FontSize'),'FontSize',16);
    saveas(fig11, fullfile(method_dir, ['UMAP_Embedding_perLatentVariable' file_suffix '.png']));
    %% Plot 2: Time Domain Reconstruction (Test Set)
    % Zero-lag correlation
    Z_true = h_test; 
    Z_recon = h_rec_test_final; % Use non-normalized for correlation calc
    maxLag = 200; lags = -maxLag:maxLag;
    zeroLagCorr = zeros(1, param.N_F);
    for f = 1:param.N_F
        c = xcorr(Z_true(:,f), Z_recon(:,f), maxLag, 'coeff');
        zeroLagCorr(f) = c(lags==0);
    end
      
    plotTimeDomainReconstruction(h_test, h_rec_test_final, param, 'UMAP', num_sig_components, zeroLagCorr, method_dir);
    %% Plot 4: Band Power Bar Chart & FFT
    % Setup FFT
    N = size(h_test, 1);
    L = round(1 * fs_new); % 1 sec window
    nTrials = floor(N/L);
    f_freq = (0:L-1)*(fs_new/L);
    nHz = L/2+1;
    f_plot = f_freq(1:nHz);
    
    Ht = zeros(L, param.N_F, nTrials);
    Hr = zeros(L, param.N_F, nTrials);
    R2_trials = zeros(L, param.N_F, nTrials);
    
    for tr = 1:nTrials
        idx = (tr-1)*L + (1:L);
        Ht(:,:,tr) = fft(h_test(idx, :));
        Hr(:,:,tr) = fft(h_rec_test_final(idx, :));
        for fidx = 1:param.N_F
            num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
            den = abs(Ht(:,fidx,tr)).^2 + eps;
            R2_trials(:,fidx,tr) = 1 - num./den;
        end
    end
    R2_avg = mean(R2_trials, 3);
    Ht_avg = mean(Ht, 3);
    Hr_avg = mean(Hr, 3);
    
    % Band Calc
    bands = struct('delta', [1 4], 'theta', [4 8], 'alpha', [8 13], 'beta', [13 30], 'gamma', [30 50]);
    band_names = fieldnames(bands);
    nBands = numel(band_names);
    band_avg_R2 = zeros(nBands, param.N_F);
    for b = 1:nBands
        f_range = bands.(band_names{b});
        idx = f_freq >= f_range(1) & f_freq <= f_range(2);
        for fidx = 1:param.N_F
            band_avg_R2(b, fidx) = mean(R2_avg(idx, fidx));
        end
    end
    
    fig3 = figure('Position',[50 50 1000 300]);
    bar(band_avg_R2');
    set(gca, 'XTickLabel', arrayfun(@(i) sprintf('Z_{%s}', num2str(param.f_peak(i))), 1:param.N_F, 'UniformOutput', false));
    ylim([-1 1]); legend(band_names, 'Location', 'southeastoutside');
    ylabel('Mean R^2'); xlabel('Latent');
    title(['UMAP Band-wise R^2 (Test Set , k=' num2str(num_sig_components) ')']); grid on;
    set(findall(fig3,'-property','FontSize'),'FontSize',20);
    saveas(fig3, fullfile(method_dir, ['UMAP_Bandwise_R2' file_suffix '.png']));
    
    
    %% Plot 5: Coherence Analysis (Chronux)
    % Multitaper params
    params_coh.Fs = fs_new; 
    params_coh.tapers = [3 5]; 
    params_coh.pad = 0;
    params_coh.err = [0 0]; % No error bars for speed in heatmap
    
    movingwin = [1 0.05]; % 1s window, 50ms step
    
    fig4 = figure('Position',[50 50 1000 600]);
    tiledlayout(2, ceil(param.N_F/2), 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['UMAP Coherence Analysis (Test Set, k=' num2str(num_sig_components) ')']);
    
    for i = 1:param.N_F
        nexttile;
        try
            [C,~,~,~,~,t_coh,f_coh] = cohgramc(h_train(:, i), h_rec_train_final(:, i), movingwin, params_coh);
            imagesc(t_coh, f_coh, C'); axis xy;
            xlabel('Time (s)'); ylabel('Freq (Hz)');
            caxis([0 1]); colorbar;
        catch
            text(0.5,0.5,'Chronux Toolbox missing or error','HorizontalAlignment','center');
        end
        title(['Latent ' num2str(i)]);
    end
    set(findall(fig4,'-property','FontSize'),'FontSize',20);
    saveas(fig4, fullfile(method_dir, ['UMAP_Coherence' file_suffix '.png']));
    
    
    %% Plot 6: Scatter Mean Band Amplitudes
    Ht_amp = abs(Ht_avg(1:nHz, :)); Hr_amp = abs(Hr_avg(1:nHz, :));
    Ht_amp = Ht_amp ./ max(Ht_amp(:)); Hr_amp = Hr_amp ./ max(Hr_amp(:));
    
    mean_true = zeros(nBands, param.N_F); mean_recon = zeros(nBands, param.N_F);
    std_true = zeros(nBands, param.N_F); std_recon = zeros(nBands, param.N_F);
    
    for b = 1:nBands
        idx = f_plot >= bands.(band_names{b})(1) & f_plot <= bands.(band_names{b})(2);
        mean_true(b,:) = mean(Ht_amp(idx,:), 1); mean_recon(b,:) = mean(Hr_amp(idx,:), 1);
        std_true(b,:) = std(Ht_amp(idx,:), 0, 1); std_recon(b,:) = std(Hr_amp(idx,:), 0, 1);
    end
    flat_true = mean_true(:); flat_recon = mean_recon(:);
    band_lbls = repelem(band_names, param.N_F);
    
    fig5 = figure('Position',[50 50 1400 300]);
    tiledlayout(1, nBands, 'TileSpacing', 'loose', 'Padding', 'compact');
    sgtitle(['UMAP Band Mean FFT Amplitudes (k=' num2str(num_sig_components) ')']);
    colors = lines(nBands); markers = {'o','s','d','h','^','hexagram','<','>'};
    
    for b = 1:nBands    
        nexttile; hold on;
        idx_b = strcmp(band_lbls, band_names{b});
        x = flat_true(idx_b); y = flat_recon(idx_b);
        
        for m = 1:length(markers)
            if m>numel(x), break; end
            scatter(x(m), y(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
            errorbar(x(m), y(m), std_true(b,m), std_recon(b,m), 'LineStyle', 'none', 'Color', colors(b,:));
        end
        plot([min(x) max(x)], [min(x) max(x)], 'Color', colors(b,:), 'LineWidth', 2);
        R = corrcoef(x,y); text(mean(x), mean(y), sprintf('R^2=%.2f', R(1,2)^2), 'Color', colors(b,:));
        title(band_names{b}); grid on;
    end
    set(findall(fig5,'-property','FontSize'),'FontSize',20);
    saveas(fig5, fullfile(method_dir, ['UMAP_Scatter_Mean' file_suffix '.png']));
    
    
    %% Plot 7: Scatter Per-Trial
    
    plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, num_sig_components, "UMAP", method_dir);
end
%% Output Structure
outUMAP = struct();
outUMAP.umap_train = umap_train;
% outUMAP.umap_test = umap_test;
outUMAP.h_recon_test = h_rec_test_final;
outUMAP.h_recon_train = h_rec_train_final;
outUMAP.MSE_test_curve = MSE_test_curve;
outUMAP.R2_test_curve = R2_test_curve;
outUMAP.MSE_train_curve = MSE_train_curve;
outUMAP.R2_train_curve = R2_train_curve;
outUMAP.MSE_test_global = MSE_test_global; 
outUMAP.R2_test_global = R2_test_global;
outUMAP.MSE_train_global = MSE_train_global; 
outUMAP.R2_train_global = R2_train_global;
outUMAP.n_neighbors = n_neighbors;
outUMAP.min_dist = min_dist;
outUMAP.corr_UMAP = corr_UMAP;
outUMAP.R_full = R_UMAP; 

close All;
end