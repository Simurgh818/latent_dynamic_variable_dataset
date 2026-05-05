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
% Reverting back to standard spatial features (Channels x Time -> Time x Channels)
eeg_train = double(s_train)';
eeg_test  = double(s_test)';

% We no longer need to chop off the last time point
% h_train and h_test remain exactly as they were passed in
rng(42,'twister');

%% 4. Run UMAP (Train) & Transform (Test)
disp(['Running UMAP (k=' num2str(num_sig_components) ') on Training Set...']);

% WORKAROUND 1: The Meehan run_umap library requires n_components >= 2.
umap_calc_components = max(2, num_sig_components);

% 1. Compute PCA for initialization
[~, score] = pca(eeg_train); 
init_coords = score(:, 1:umap_calc_components);
[~, score_test] = pca(eeg_test); 
init_coords_test = score_test(:, 1:umap_calc_components);

% 2. Normalize 
init_coords = 10 * (init_coords / max(abs(init_coords(:))));

% Run UMAP on training data 
try
    [umap_train_raw, umap_struct] = run_umap(eeg_train, ...
        'n_neighbors', n_neighbors, ...
        'min_dist', min_dist, ...
        'n_components', umap_calc_components,...        
        'method', 'MEX', ...              
        'verbose', 'none', ...
        'metric', 'euclidean', ...
        'randomize', true, ...
        'init', init_coords, ...      
        'save_template_file', false);    
catch ME
    warning('UMAP Train failed: %s. Attempting fallback settings.', ME.message);
    rethrow(ME); 
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
        'init', init_coords_test);              
catch
    warning('Could not project test data via template. Running UMAP independently on Test.');
    umap_test_raw = run_umap(eeg_test, ...
        'n_neighbors', n_neighbors, ...
        'min_dist', min_dist, ...
        'n_components', umap_calc_components, ...
        'method', 'MEX', ...   
        'metric', 'euclidean', ...
        'randomize', true, ...
        'init', init_coords_test);              
end

% Extract only the requested number of components for analysis
umap_train = umap_train_raw(:, 1:num_sig_components);
umap_test  = umap_test_raw(:, 1:num_sig_components);

% Mapping components to latents
C = umap_train;   % T x nComp
H = h_train(1:size(C,1), :);
[corr_UMAP, R_UMAP] = match_components_to_latents(C, H, 'UMAP',num_sig_components);

%% 5. Reconstruction Loop (Train Mapping -> Test Eval)
% 1. Learn Map: UMAP_train -> H_train
W_k_train = umap_train_raw \ h_train;
W_k = umap_test_raw \ h_test;

% 2. Apply Map to Test: UMAP_test * W -> H_rec_test
h_rec_train_final = umap_train_raw * W_k_train;
h_rec_test_final = umap_test_raw * W_k;

fs_new = param.fs;

%% 6. Zero-Lag Correlation (Moved to mathematical block)
Corr = zeros(1, param.N_F);
for f = 1:param.N_F
    c  = corrcoef(h_test(:,f), h_rec_test_final(:,f));
    Corr(f) = c(1,2);
end

%% 7. Frequency & Spectral R2 Math (Runs on ALL workers!)
T = size(h_test, 1);
trial_dur = 1; 
L = round(trial_dur * param.fs);
nTrials = floor(T/L);
f_axis = (0:L-1)*(param.fs/L);
nHz = floor(L/2) + 1;
f_plot = f_axis(1:nHz);

Ht = zeros(L, param.N_F, nTrials);
Hr = zeros(L, param.N_F, nTrials);
R2_trials = zeros(L, param.N_F, nTrials);

% --- FFT Calculation ---
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

Ht_avg = mean(abs(Ht(1:nHz, :, :)), 3);
Hr_avg = mean(abs(Hr(1:nHz, :, :)), 3);
R2_avg = mean(R2_trials, 3);

% --- Spectral R2 Calculation ---
bands = struct('delta',[1 4], 'theta',[4 8], 'alpha',[8 13], 'beta',[13 30], 'gamma',[30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);
UMAP_R2_scores = nan(param.N_F, 1);

Ht_amp = abs(Ht(1:nHz,:,:));
Hr_amp = abs(Hr(1:nHz,:,:));

max_t = max(Ht_amp(:)); if max_t==0, max_t=1; end
max_r = max(Hr_amp(:)); if max_r==0, max_r=1; end
Ht_amp = Ht_amp ./ max_t; Hr_amp = Hr_amp ./ max_r;

true_vals = cell(nBands,1);
recon_vals = cell(nBands,1);

for b = 1:nBands
    f_range  = bands.(band_names{b});
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
    
    true_vals{b}  = squeeze(mean(Ht_amp(idx_band,:,:), 1, 'omitnan'));
    recon_vals{b} = squeeze(mean(Hr_amp(idx_band,:,:), 1, 'omitnan'));
    
    if b == 4, target_zs = [4, 5];
    elseif b == 5, target_zs = 6;
    else, target_zs = b; end
    
    for z = 1:param.N_F
        if ismember(z, target_zs)
            x_z = true_vals{b}(z,:);
            y_z = recon_vals{b}(z,:);
            R_coef = corrcoef(x_z, y_z);
            if numel(R_coef) > 1, r_sq = R_coef(1,2)^2; else, r_sq = 0; end
            UMAP_R2_scores(z) = r_sq;
        end
    end
end

%% ============================================================
% PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if isempty(getCurrentTask())
    
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
    fig11 = figure('Position', [100, 100, 1400, 900], 'Visible', 'off');
    t = tiledlayout(ceil(n_latents/3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for i = 1:n_latents
        nexttile;
        scatter(umap_test_raw(:,1), umap_test_raw(:,2), 15, h_test(:,i), 'filled');
        xlim([-10 10]); xticks(-10:10:10);
        
        % FIX: Using the newly calculated Pearson Corr instead of the deleted R2
        rho = round(Corr(i), 2);
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
    plotTimeDomainReconstruction(h_test, h_rec_test_final, param, 'UMAP', num_sig_components, Corr, method_dir);
    
    % Embedding Traces:
    plotCTraces(num_sig_components, param, umap_test_raw, method_dir, file_suffix);
    
    %% Plot 4: Band Power Bar Chart & FFT     
    save_path_fft = fullfile(method_dir, ['UMAP_FFT_True_vs_Recon' file_suffix '.png']);
    plotFrequencySpectra(Ht_avg, Hr_avg, f_plot, 'UMAP', param, num_sig_components, save_path_fft);
    
    br2_path = fullfile(method_dir, ['UMAP_Bandwise_R2' file_suffix '.png']);
    plotBandwiseR2(R2_avg, f_axis, param, num_sig_components, 'UMAP', br2_path);
    
    %% Plot 5: Coherence Analysis (Chronux)
    params_coh.Fs = fs_new; 
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
    fig5 = figure('Position',[50 50 1400 300], 'Visible', 'off');
    tiledlayout(1, nBands, 'TileSpacing', 'loose', 'Padding', 'compact');
    sgtitle(['UMAP Band Mean FFT Amplitudes (k=' num2str(num_sig_components) ')']);
    colors = lines(nBands); markers = {'o','s','d','h','^'};
    
    for b = 1:nBands    
        nexttile; hold on;
        x = mean(true_vals{b}, 2, 'omitnan');
        y = mean(recon_vals{b}, 2, 'omitnan');
        std_x = std(true_vals{b}, 0, 2, 'omitnan');
        std_y = std(recon_vals{b}, 0, 2, 'omitnan');
        
        for m = 1:length(markers)
            if m>numel(x), break; end
            scatter(x(m), y(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
            errorbar(x(m), y(m), std_x(m), std_y(m), 'LineStyle', 'none', 'Color', colors(b,:));
        end
        plot([min(x) max(x)], [min(x) max(x)], 'Color', colors(b,:), 'LineWidth', 2);
        R = corrcoef(x,y); text(mean(x), mean(y), sprintf('R^2=%.2f', R(1,2)^2), 'Color', colors(b,:));
        title(band_names{b}); grid on;
    end
    set(findall(fig5,'-property','FontSize'),'FontSize',20);
    saveas(fig5, fullfile(method_dir, ['UMAP_Scatter_Mean' file_suffix '.png']));
    close(fig5);
    
    %% Plot 7: Scatter Per-Trial
    plotBandScatterPerTrial(true_vals, recon_vals, UMAP_R2_scores, band_names, param, num_sig_components, "UMAP", method_dir);
end

%% Output Structure (FIX: Removed all deleted curve/global variables)
outUMAP = struct();
outUMAP.umap_train = umap_train;
outUMAP.h_recon_test = h_rec_test_final;
outUMAP.h_recon_train = h_rec_train_final;
outUMAP.n_neighbors = n_neighbors;
outUMAP.min_dist = min_dist;
outUMAP.corr_UMAP = corr_UMAP;
outUMAP.R_full = R_UMAP; 
outUMAP.Corr = Corr; % <--- Now guaranteed to exist!
outUMAP.spectral_R2 = UMAP_R2_scores; % <--- Now guaranteed to exist!
close All;

end