%% MAIN SCRIPT FOR DIMENSIONALITY REDUCTION BENCHMARK
clear; clc; close all;
%% ----------------------------------------------------------
% 1. Load & Prepare Data
% ----------------------------------------------------------
% Paths
if exist('I:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'simEEG']; 
    baseFolder = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'Results'];
elseif exist('H:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'simEEG']; 
    baseFolder = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'Results'];
elseif exist('G:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'simEEG'];
    baseFolder = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'Results'];
elseif ismac && exist('/Users/asederberg6/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology', 'dir')
    one_drive_dir = '/Users/asederberg6/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology';
    path_to_files = '/Users/asederberg6/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Dabiri, Sina''s files - Dr. Sederberg MaTRIX Lab';
    input_dir = [path_to_files filesep ...
        'Shared Code' filesep 'simEEG'];
    baseFolder = [path_to_files filesep ...
        'Dimensionality Reduction Review Paper'];    
else
    error('Unknown system: Cannot determine input and output paths.');
end

%% Loop through experiments
conditions = {'set4'}; %,'ou', 'set2',  linear, nonlinear
nDatasets  = 10; % 10 datasets
k_range    = 1:10; % 10 k components
nK         = numel(k_range);

% Store results: structure indexed by method name
methods = {'PCA', 'AE','ICA'};  % 'PCA', 'AE','ICA'
% --- Define Marker & Line Styles for distinct plotting ---
method_markers = {'o', 's', '^', 'd', 'v', 'p'}; % Circle, Square, Triangle, Diamond, etc.
method_lines = {'-', '--', '-.', ':', '-', '--'}; % Solid, Dashed, Dash-Dot, Dotted, etc.

EXP = struct();
param = struct();
param.duration = [1000];

RESULTS = struct();
RESULTS.meta = struct();
RESULTS.meta.created = datetime;
RESULTS.meta.description = "Dimensionality reduction benchmark (80:20 Split)";

% target_workers = 5; 
% current_pool = gcp('nocreate');
% if isempty(current_pool)
%     parpool(target_workers);
% elseif current_pool.NumWorkers ~= target_workers
%     delete(current_pool);
%     parpool(target_workers);
% end

%% Loop through experiments
for c = 1:numel(conditions)
    cond = conditions{c};
    fprintf('\n=== Running condition: %s ===\n', cond);
    
    % Preallocate temporary storage for parallel workers
    dataset_results = cell(1, nDatasets);
    
   % ---------------------------------------------------------------------
    % PARALLEL LOOP
    % ---------------------------------------------------------------------
    for d = 1:nDatasets %parfor
        fprintf('Dataset %d / %d (Worker Processing)\n', d, nDatasets);
        data = struct();
        
        % --- 1. Determine Filename (Local to Worker) ---
        if d < 10 
            eegFilename  = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, param.duration(1));
        else
            eegFilename  = sprintf('simEEG_%s_spat%d_dur%d', cond, d, param.duration(1));
        end
        dataset_name = eegFilename;
        
        % --- 2. Load Single Continuous Dataset ---
        loader = load(fullfile(input_dir, [eegFilename '.mat']));
        s_eeg_all   = double(loader.sim_eeg_vals);
        h_f_all     = double(loader.all_h_F');
        
        % Recalculate parameters locally
        local_param = loader.param; 
        fs          = 1 / loader.dt;
        local_param.fs = fs;
        
        % Determine local results directory
        subfolderName = ['results_' eegFilename];
        local_results_dir = fullfile(baseFolder, subfolderName);
        if ~exist(local_results_dir, 'dir')
            mkdir(local_results_dir);
        end
        
        % --- 3. Strict 80:20 Contiguous Temporal Split ---
        % Sequential partitioning ensures spatial geometry remains constant 
        % while evaluating out-of-sample generalizability forward in time.
        idx_split = floor(0.8 * size(s_eeg_all, 2));
        
        eeg_train = s_eeg_all(:, 1:idx_split);
        eeg_test  = s_eeg_all(:, idx_split+1:end);
        
        h_f_train = h_f_all(1:idx_split, :);
        h_f_test  = h_f_all(idx_split+1:end, :);
        
        % --- 4. Independent Partition Normalization ---
        % Normalize latents to zero-mean, unit-variance computed strictly within 
        % respective partitions to prevent statistical scaling leakage.
        H_train = h_f_train ./ std(h_f_train, 0, 1);
        H_test  = h_f_test ./ std(h_f_test, 0, 1);   
       
        data.eeg_train = eeg_train;
        data.eeg_test  = eeg_test;
        data.H_train   = H_train;
        data.H_test    = H_test;
        data.f_peak    = local_param.f_peak;
        
        % --- Method Loop ---
        dataset_res = struct();
        dataset_res.f_peak = local_param.f_peak;
        
        % Initialize a table to collect ALL entries for this dataset
        all_d_entries = table();
        
        for m = 1:numel(methods)
            method = methods{m};
            method_dir = fullfile(local_results_dir, method);
            if ~exist(method_dir, 'dir'), mkdir(method_dir); end
            
            dataset_res.(method).Comp_latent_matching_corr = cell(1, nK);
            dataset_res.(method).Comp_latent_matching_matrix = cell(1, nK);
            dataset_res.(method).direct_Component_Corr = nan(local_param.N_F, nK);
            dataset_res.(method).spectral_R2 = nan(local_param.N_F, nK);
            dataset_res.(method).matched_R2 = nan(local_param.N_F, nK);

            for ki = 1:nK
                k = k_range(ki);
                fprintf('   -> Running %s with k = %d ...\n', method, k);
                entry = runDimRedMethod( ...
                    method, data, local_param, k, ki, cond, dataset_name, method_dir,...
                    local_results_dir);
                
                % Store basic stats                           
                dataset_res.(method).Comp_latent_matching_corr{ki} = entry.Comp_latent_matching_corr;
                dataset_res.(method).Comp_latent_matching_matrix{ki} = entry.Comp_latent_matching_matrix;
                dataset_res.(method).direct_Component_Corr(:,ki) = entry.direct_Component_Corr;
                dataset_res.(method).spectral_R2(:,ki) = entry.spectral_R2;
                dataset_res.(method).matched_R2(:,ki) = entry.matched_R2;

                if ki == nK
                    dataset_res.(method).h_recon_test = entry.out.h_recon_test;
                end
                
                % --- Fix for Duplicate Variable Names ---
                current_corr_table = entry.Comp_latent_matching_corr; 
                
                if ~isempty(current_corr_table)
                    nRows = height(current_corr_table);
                    current_vars = current_corr_table.Properties.VariableNames;
                    
                    % Only add columns if they DON'T exist yet
                    if ~ismember('method', current_vars)
                        current_corr_table.method = repmat(categorical(cellstr(method)), nRows, 1);
                    end
                    if ~ismember('dataset', current_vars)
                        current_corr_table.dataset = repmat(d, nRows, 1);
                    end
                    if ~ismember('condition', current_vars)
                        current_corr_table.condition = repmat(categorical(cellstr(cond)), nRows, 1);
                    end
                    if ~ismember('k', current_vars)
                        current_corr_table.k = repmat(k, nRows, 1);
                    end
                    
                    % Append to the main collector
                    all_d_entries = [all_d_entries; current_corr_table];
                end
            end
        end
        
        % saving out snippets of data 
        snippet_seconds = 10;
        snippet_samples = min(size(data.H_test, 1), round(snippet_seconds * local_param.fs));
        time_idx = 1:snippet_samples;
        
        snippets = struct();
        snippets.time_vector = (time_idx - 1) / local_param.fs;
        snippets.H_test = data.H_test(time_idx, :); % Ground truth
        
        snippets.eeg_train = data.eeg_train; 
        snippets.param = local_param;
        
        % Collect the reconstruction from each method at the max k
        for m = 1:numel(methods)
            method = methods{m};
            % Ensure we only take the snippet window
            snippets.(method).h_recon_test = dataset_res.(method).h_recon_test(time_idx, :);
        end     
        
        % Pack variables into a structure to save cleanly
        ds_out = struct();
        ds_out.analysis = dataset_res;
        ds_out.entries  = all_d_entries;
        ds_out.param    = local_param;
        ds_out.dataset  = d;
        ds_out.cond     = cond;
        ds_out.snippet  = snippets;
        
        % Create filename (e.g., "Results_simEEG_set4_spat01.mat")
        ds_filename = fullfile(local_results_dir, sprintf('Results_%s.mat', dataset_name));
        
        % Call helper function to save (avoids parfor transparency error)
        parsave_struct(ds_filename, ds_out);
        
        % Save results for this dataset
        dataset_results{d}.(cond).output_dir = local_results_dir;
        dataset_results{d}.(cond).analysis = dataset_res;
        dataset_results{d}.(cond).entries = all_d_entries;
        dataset_results{d}.(cond).snippets = snippets;
    end % End Parfor
    
    RESULTS.data = struct();
    
    % ---------------------------------------------------------------------
    % POST-PROCESSING & PLOTTING (Serial)
    % ---------------------------------------------------------------------
    % Recover f_peak safely from any completed dataset
    for d = 1:nDatasets
        if ~isempty(dataset_results{d}) && ...
           isfield(dataset_results{d}.(cond), 'analysis') && ...
           isfield(dataset_results{d}.(cond).analysis, 'f_peak')
    
            param.f_peak = dataset_results{d}.(cond).analysis.f_peak;
            break
        end
    end
    
    % Unpack results back into EXP structure and Generate Plots
    for d = 1:nDatasets
        EXP.(cond).dataset(d) = dataset_results{d}.(cond);
        local_results_dir = EXP.(cond).dataset(d).output_dir;
        
        % Store hierarchically as RESULTS.data.set4.dataset_1, etc.
        ds_field = sprintf('dataset_%d', d);
        RESULTS.data.(cond).(ds_field).analysis = dataset_results{d}.(cond).analysis;
        RESULTS.data.(cond).(ds_field).entries  = dataset_results{d}.(cond).entries;
        RESULTS.data.(cond).(ds_field).path     = dataset_results{d}.(cond).output_dir;
        RESULTS.data.(cond).(ds_field).snippets = dataset_results{d}.(cond).snippets;
        
        for m = 1:numel(methods)
            method = methods{m};
            method_dir = fullfile(local_results_dir, method);
            
            % Generate Heatmaps (now safe in serial)
            for ki = 1:nK
                k = k_range(ki);
                R = EXP.(cond).dataset(d).analysis.(method).Comp_latent_matching_matrix{ki};
                
                if isempty(R), continue; end
                
                fig = figure('Visible', 'off'); % 
                imagesc(R);
                clim([-1 1]); 
                axis square;
                xlabel('Component index (C)'); ylabel('True latent index (h_f)');
                title(sprintf('%s: Latent–Component Correlation', method));
                colormap(parula); colorbar;
                saveas(fig, fullfile(method_dir, sprintf('%s_CorrHeatmap_k%d.png', method, k)));
                close(fig);
            end
        end
    end
end

STATS = struct();
for c = 1:numel(conditions)
    cond = conditions{c};
    for m = 1:numel(methods)
        method = methods{m};
        spectral_R2_all = nan(nDatasets, length(param.f_peak), nK);
        direct_Component_Corr_all = nan(nDatasets, length(param.f_peak), nK);
        matched_R2_all = nan(nDatasets, length(param.f_peak), nK);

        for d = 1:nDatasets
            direct_Component_Corr_all(d, :, :) = EXP.(cond).dataset(d).analysis.(method).direct_Component_Corr;
            spectral_R2_all(d, :, :) = EXP.(cond).dataset(d).analysis.(method).spectral_R2;
            matched_R2_all(d, :, :) = EXP.(cond).dataset(d).analysis.(method).matched_R2; % <-- NEW
        end
        % Per-latent detailed stats (Mean and Std across datasets)
        STATS.(cond).(method).direct_Component_Corr.mean  = squeeze(mean(direct_Component_Corr_all, 1, 'omitnan'));
        STATS.(cond).(method).direct_Component_Corr.std   = squeeze(std(direct_Component_Corr_all, 0, 1, 'omitnan'));
        STATS.(cond).(method).spectral_R2.mean = squeeze(mean(spectral_R2_all, 1, 'omitnan'));
        STATS.(cond).(method).spectral_R2.std  = squeeze(std(spectral_R2_all, 0, 1, 'omitnan'));
        STATS.(cond).(method).matched_R2.mean = squeeze(mean(matched_R2_all, 1, 'omitnan'));
        STATS.(cond).(method).matched_R2.std  = squeeze(std(matched_R2_all, 0, 1, 'omitnan'));
    end
end

CORR_STATS = struct();
for c = 1:numel(conditions)
    cond = conditions{c};
    for m = 1:numel(methods)
        method = methods{m};
        for ki = 1:nK
            % Collect corr tables from all datasets
            corr_mat = nan(nDatasets, size(param.f_peak,2));
            for d = 1:nDatasets
                tbl = EXP.(cond).dataset(d).analysis.(method).Comp_latent_matching_corr{ki};
                if isempty(tbl)
                    continue
                end
                corr_mat(d, tbl.h_f) = tbl.corr_value;
            end
            CORR_STATS.(cond).(method)(ki).mean = mean(corr_mat,1,'omitnan');
            CORR_STATS.(cond).(method)(ki).std  = std(corr_mat,0,1,'omitnan');
        end
    end
end

for c = 1:numel(conditions)
    cond = conditions{c};
    for ki = 1:nK
        k = k_range(ki);
        fig0 = figure; hold on;
        colors = lines(numel(methods));
        for m = 1:numel(methods)
            method = methods{m};
            mu = CORR_STATS.(cond).(method)(ki).mean;
            sd = CORR_STATS.(cond).(method)(ki).std;
            errorbar(param.f_peak, mu, sd, '-o', ...
                'LineWidth',1.8, ...
                'Color',colors(m,:), ...
                'DisplayName',method);
        end
        xlabel('Peak Frequency (Hz)');
        ylabel('Matched Correlation');
        xticks(param.f_peak);
        ylim([0 1]);
        title(sprintf('Latent Variable–Component Corr (k=%d)', k));
        grid on;
        legend('Location','eastoutside');
        set(gca, 'FontSize', 16);
        corr_figure_name = fullfile(local_results_dir, sprintf('Latent Variable–Component Corr %s k_%d.png', cond, k));
        saveas(fig0, corr_figure_name);
    end
end

% In the post-processing section:
all_corr_tables = table();
for d = 1:nDatasets
    if isfield(dataset_results{d}, cond) && ...
       isfield(dataset_results{d}.(cond), 'entries')
        
        % Just stack the already-complete tables
        all_corr_tables = [all_corr_tables; dataset_results{d}.(cond).entries];
    end
end

if isempty(all_corr_tables)
    warning('Correlation table is empty! Check method outputs.');
    summary = table();
    summary_min = table();
    good_counts = table();
    threshold = 0.39;
else
    summary = groupsummary(all_corr_tables, 'method', 'mean', 'corr_value');
    summary_min = groupsummary(all_corr_tables, 'method', 'min', 'corr_value');
    threshold = 0.39; % since UMAP best performance is 0.4
    good_counts = groupsummary( ...
                    all_corr_tables(all_corr_tables.corr_value > threshold,:), ...
                    'method',@sum,'corr_value');
end

results.summary.mean = summary;
results.summary.min  = summary_min;
results.summary.good_counts = good_counts;
results.summary.threshold = threshold;

% Save file -------------------------------------------------------------
save(fullfile(local_results_dir, "RESULTS.mat"), "RESULTS", "-v7.3");
save(fullfile(local_results_dir, "Component_latentVariable_Corr.mat"), ...
     "all_corr_tables", "results");

%% ----------------------------------------------------------
% COLORBLIND PALETTE & DUAL-CODING SETUP
% ----------------------------------------------------------
% Okabe-Ito Colorblind-Safe Palette
cb_palette = [
    0.00, 0.45, 0.70;  % Blue
    0.90, 0.60, 0.00;  % Orange
    0.00, 0.60, 0.50;  % Bluish Green
    0.80, 0.40, 0.00;  % Vermillion
    0.35, 0.70, 0.90;  % Sky Blue
    0.80, 0.60, 0.70;  % Reddish Purple
    0.95, 0.90, 0.25;  % Yellow
    0.20, 0.20, 0.20   % Dark Gray
];
% Dynamically map methods to the Okabe-Ito palette
colors = cb_palette(mod(0:numel(methods)-1, size(cb_palette, 1)) + 1, :);

% Dual coding markers and linestyles
method_markers = {'o', 's', '^', 'd', 'v', 'p'};
method_lines   = {'-', '--', '-.', ':', '-', '--'};

%% ----------------------------------------------------------
% 4. Plot Mean Latent Correlation vs # components (k)
% ----------------------------------------------------------
% FIGURE 1: DIRECT TIME-DOMAIN CORRELATION
fig1 = figure('Position', [100, 100, 1000, 600], 'Visible', 'off');
hold on;
set(gca, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20, 'FontWeight', 'bold');
for c = 1:numel(conditions)
    cond = conditions{c};
    
    for m = 1:numel(methods)
        method = methods{m};
        m_idx = mod(m-1, length(method_markers)) + 1;
        l_idx = mod(m-1, length(method_lines)) + 1;
        
        mu_k = mean(STATS.(cond).(method).direct_Component_Corr.mean, 1, 'omitnan');
        sd_k = std(STATS.(cond).(method).direct_Component_Corr.mean, 0, 1, 'omitnan');
        
        % Blend color with white to simulate 0.6 MarkerFaceAlpha on ErrorBar
        face_col = 0.6 * colors(m, :) + 0.4 * [1, 1, 1];
        
        errorbar(k_range, mu_k, sd_k, ...
            'LineStyle', method_lines{l_idx}, ...
            'Marker', method_markers{m_idx}, ...
            'LineWidth', 2.5, ...
            'MarkerSize', 8, ...  
            'Color', colors(m, :), ...
            'MarkerFaceColor', face_col, ...
            'DisplayName', method);
    end
end
xlabel('Number of Components (k)', 'FontSize', 22, 'FontWeight', 'bold');
xticks(k_range);
ylabel('Mean Correlation ($Z$ and $\mathbf{\hat{z}}$)', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'bold');
ylim([0 1]);
title('Mean Latent Correlation vs. k', 'FontSize', 24, 'FontWeight', 'bold');
grid on;
lgd1 = legend('Location', 'eastoutside');
lgd1.FontSize = 18;
summary_trace_name = fullfile(local_results_dir, 'Main_Summary_Corr_vs_k.png');
exportgraphics(fig1, summary_trace_name, 'Resolution', 600);
close(fig1);

% ==========================================================
% FIGURE 1b: TIME-DOMAIN MATCHED R^2
% ==========================================================
fig1b = figure('Position', [125, 125, 1000, 600], 'Visible', 'off');
hold on;
set(gca, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20, 'FontWeight', 'bold');
for c = 1:numel(conditions)
    cond = conditions{c};
    
    for m = 1:numel(methods)
        method = methods{m};
        m_idx = mod(m-1, length(method_markers)) + 1;
        l_idx = mod(m-1, length(method_lines)) + 1;
        
        % Average the True Time-Domain R^2 across all latents 
        mu_k_r2 = mean(STATS.(cond).(method).matched_R2.mean, 1, 'omitnan');
        sd_k_r2 = std(STATS.(cond).(method).matched_R2.mean, 0, 1, 'omitnan');
        
        face_col = 0.6 * colors(m, :) + 0.4 * [1, 1, 1];
        
        errorbar(k_range, mu_k_r2, sd_k_r2, ...
            'LineStyle', method_lines{l_idx}, ...
            'Marker', method_markers{m_idx}, ...
            'LineWidth', 2.5, ...
            'MarkerSize', 8, ...  
            'Color', colors(m, :), ...
            'MarkerFaceColor', face_col, ...
            'DisplayName', method);
    end
end
xlabel('Number of Components (k)', 'FontSize', 22, 'FontWeight', 'bold');
xticks(k_range); 
ylabel('Mean Time-Domain R^2', 'FontSize', 22, 'FontWeight', 'bold');
ylim([0 1]); 
title('Time-Domain R^2 vs. k', 'FontSize', 24, 'FontWeight', 'bold');
grid on;
lgd1b = legend('Location', 'eastoutside');
lgd1b.FontSize = 18;
summary_r2_name = fullfile(local_results_dir, 'Main_Summary_TimeDomain_R2_vs_k.png');
exportgraphics(fig1b, summary_r2_name, 'Resolution', 600);
close(fig1b);

% ==========================================================
% FIGURE 1c: MATCHED COMPONENT CORRELATION
% ==========================================================
fig1c = figure('Position', [150, 150, 1000, 600], 'Visible', 'off');
hold on;
set(gca, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20, 'FontWeight', 'bold');
for c = 1:numel(conditions)
    cond = conditions{c};
    
    for m = 1:numel(methods)
        method = methods{m};
        m_idx = mod(m-1, length(method_markers)) + 1;
        l_idx = mod(m-1, length(method_lines)) + 1;
        
        mu_k_matched = zeros(1, nK);
        sd_k_matched = zeros(1, nK);
        
        for ki = 1:nK
            mu_k_matched(ki) = mean(CORR_STATS.(cond).(method)(ki).mean, 'omitnan');
            sd_k_matched(ki) = std(CORR_STATS.(cond).(method)(ki).mean, 0, 'omitnan');
        end
        
        face_col = 0.6 * colors(m, :) + 0.4 * [1, 1, 1];
        
        errorbar(k_range, mu_k_matched, sd_k_matched, ...
            'LineStyle', method_lines{l_idx}, ...
            'Marker', method_markers{m_idx}, ...
            'LineWidth', 2.5, ...
            'MarkerSize', 8, ...  
            'Color', colors(m, :), ...
            'MarkerFaceColor', face_col, ...
            'DisplayName', method);
    end
end
xlabel('Number of Components (k)', 'FontSize', 22, 'FontWeight', 'bold');
xticks(k_range); 
ylabel('Mean Matched Correlation (\rho)', 'FontSize', 22, 'FontWeight', 'bold');
ylim([0 1]);
title('Mean Matched Latent Correlation vs. k', 'FontSize', 24, 'FontWeight', 'bold');
grid on;
lgd1c = legend('Location', 'eastoutside');
lgd1c.FontSize = 18;
summary_matched_name = fullfile(local_results_dir, 'Main_Summary_MatchedCorr_vs_k.png');
exportgraphics(fig1c, summary_matched_name, 'Resolution', 600);
close(fig1c);

%% ----------------------------------------------------------
% 5. Plot Corr and Spectral R^2 vs Peak Frequency (for target_k)
% ----------------------------------------------------------
target_k = 10;
ki_target = find(k_range == target_k);
if ~isempty(ki_target)
    
    % ==========================================================
    % FIGURE 2a: MATCHED CORRELATION VS. PEAK FREQUENCY
    % ==========================================================
    fig2a = figure('Position', [100, 100, 1000, 600], 'Color', 'w', 'Visible', 'off');
    hold on;
    ax2a = gca;
    ax2a.Toolbar.Visible = 'off'; % Prevents zoom/home icons from appearing in export
    set(ax2a, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20, 'FontWeight', 'bold');
    
    for c = 1:numel(conditions)
        cond = conditions{c};
        
        for m = 1:numel(methods)
            method = methods{m};
            m_idx = mod(m-1, length(method_markers)) + 1;
            l_idx = mod(m-1, length(method_lines)) + 1;
            
            mu = CORR_STATS.(cond).(method)(ki_target).mean;
            sd = CORR_STATS.(cond).(method)(ki_target).std;
            
            x_col = reshape(double(param.f_peak), [], 1);
            y_col = reshape(double(mu), [], 1);
            e_col = reshape(double(sd), [], 1);
            
            face_col = 0.6 * colors(m, :) + 0.4 * [1, 1, 1];
            
            errorbar(x_col, y_col, e_col, ...
                'LineStyle', method_lines{l_idx}, ...
                'Marker', method_markers{m_idx}, ...
                'LineWidth', 2.5, ...
                'MarkerSize', 8, ...
                'Color', colors(m, :), ...
                'MarkerFaceColor', face_col, ...
                'DisplayName', char(method));
        end
    end
    
    xlabel('Peak Frequency (Hz)', 'FontSize', 22, 'FontWeight', 'bold');
    ylabel('Matched Correlation (\rho)', 'FontSize', 22, 'FontWeight', 'bold');
    xticks(param.f_peak);
    ylim([0 1]); 
    title(sprintf('Matched Correlation vs. Frequency (k=%d)', target_k), 'FontSize', 24, 'FontWeight', 'bold');
    grid on; 
    lgd2a = legend('Location', 'eastoutside'); 
    lgd2a.FontSize = 18;
    
    summary_corr_freq_name = fullfile(local_results_dir, sprintf('Summary_MatchedCorr_vs_Freq_k%d.png', target_k));
    exportgraphics(fig2a, summary_corr_freq_name, 'Resolution', 600);
    close(fig2a);

    % ==========================================================
    % FIGURE 2b: SPECTRAL R^2 VS. PEAK FREQUENCY
    % ==========================================================
    fig2b = figure('Position', [125, 125, 1000, 600], 'Color', 'w', 'Visible', 'off');
    hold on;
    ax2b = gca;
    ax2b.Toolbar.Visible = 'off'; % Prevents zoom/home icons from appearing in export
    set(ax2b, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20, 'FontWeight', 'bold');
    
    for c = 1:numel(conditions)
        cond = conditions{c};
        
        for m = 1:numel(methods)
            method = methods{m};
            m_idx = mod(m-1, length(method_markers)) + 1;
            l_idx = mod(m-1, length(method_lines)) + 1;
            
            mu = STATS.(cond).(method).spectral_R2.mean;
            sd = STATS.(cond).(method).spectral_R2.std;
            
            if size(mu, 2) > 1
                mu = mu(:, ki_target);
                sd = sd(:, ki_target);
            end
            
            x_col = reshape(double(param.f_peak), [], 1);
            y_col = reshape(double(mu), [], 1);
            e_col = reshape(double(sd), [], 1);
            
            face_col = 0.6 * colors(m, :) + 0.4 * [1, 1, 1];
            
            errorbar(x_col, y_col, e_col, ...
                'LineStyle', method_lines{l_idx}, ...
                'Marker', method_markers{m_idx}, ...
                'LineWidth', 2.5, ...
                'MarkerSize', 8, ...
                'Color', colors(m, :), ...
                'MarkerFaceColor', face_col, ...
                'DisplayName', char(method));
        end
    end
    
    xlabel('Peak Frequency (Hz)', 'FontSize', 22, 'FontWeight', 'bold');
    ylabel('Spectral R^2', 'FontSize', 22, 'FontWeight', 'bold');
    xticks(param.f_peak);
    ylim([0 1]); 
    title(sprintf('Spectral R^2 vs. Frequency (k=%d)', target_k), 'FontSize', 24, 'FontWeight', 'bold');
    grid on; 
    lgd2b = legend('Location', 'eastoutside'); 
    lgd2b.FontSize = 18;
    
    summary_spec_freq_name = fullfile(local_results_dir, sprintf('Summary_SpectralR2_vs_Freq_k%d.png', target_k));
    exportgraphics(fig2b, summary_spec_freq_name, 'Resolution', 600);
    close(fig2b);

else
    fprintf('k=%d is not in your current k_range. Skipping the frequency profile plots.\n', target_k);
end

%% ----------------------------------------------------------
% 6. Plot Matching Correlation vs k (Per Latent Variable)
% ----------------------------------------------------------
for c = 1:numel(conditions)
    cond = conditions{c};
    
    fig3 = figure('Position', [100, 100, 1600, 900], 'Visible', 'off');
    nLatents = length(param.f_peak);
    nCols = ceil(sqrt(nLatents));
    nRows = ceil(nLatents / nCols);
    t3 = tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    title(t3, sprintf('Matching Correlation vs. k per Latent Variable (%s)', cond), ...
        'FontSize', 26, 'FontWeight', 'bold');
    
    RESULTS.stats_per_latent.(cond) = struct();
    
    for f = 1:nLatents
        nexttile; hold on;
        set(gca, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 18, 'FontWeight', 'bold');
        peak_freq = param.f_peak(f);
        
        for m = 1:numel(methods)
            method = methods{m};
            m_idx = mod(m-1, length(method_markers)) + 1;
            l_idx = mod(m-1, length(method_lines)) + 1;
            
            mu_k = zeros(1, nK);
            sd_k = zeros(1, nK);
            for ki = 1:nK
                mu_k(ki) = CORR_STATS.(cond).(method)(ki).mean(f);
                sd_k(ki) = CORR_STATS.(cond).(method)(ki).std(f);
            end
            
            face_col = 0.6 * colors(m, :) + 0.4 * [1, 1, 1];
            
            errorbar(k_range, mu_k, sd_k, ...
                'LineStyle', method_lines{l_idx}, ...
                'Marker', method_markers{m_idx}, ...
                'LineWidth', 2.5, ...
                'MarkerSize', 8, ...
                'Color', colors(m, :), ...
                'MarkerFaceColor', face_col, ...
                'DisplayName', method);
            
            latent_name = sprintf('Z_%gHz', peak_freq);
            latent_name = strrep(latent_name, '.', '_'); 
            
            RESULTS.stats_per_latent.(cond).(method).(latent_name).k = k_range;
            RESULTS.stats_per_latent.(cond).(method).(latent_name).corr_mean = mu_k;
            RESULTS.stats_per_latent.(cond).(method).(latent_name).corr_std = sd_k;
        end
        
        xlabel('Number of Components (k)', 'FontSize', 20, 'FontWeight', 'bold');
        ylabel('Matching Correlation (\rho)', 'FontSize', 20, 'FontWeight', 'bold');
        xticks(k_range);
        if numel(k_range) > 1
            xlim([min(k_range), max(k_range)]);
        end
        ylim([0 1]);
        title(sprintf('Latent %d: %g Hz', f, peak_freq), 'FontSize', 22, 'FontWeight', 'bold');
        grid on;
        
        if f == 1
            lgd3 = legend('Location', 'best');
            lgd3.FontSize = 16;
        end
    end
    
    fig_name = fullfile(local_results_dir, sprintf('Matching_Corr_vs_k_Per_Latent_%s.png', cond));
    exportgraphics(fig3, fig_name, 'Resolution', 600);
    close(fig3);
end

% Re-save RESULTS.mat to capture the newly added stats_per_latent structure
save(fullfile(local_results_dir, "RESULTS.mat"), "RESULTS", "-v7.3");
%% ---------------------------------------------------------------------
%  HELPER FUNCTIONS
% ---------------------------------------------------------------------
function parsave_struct(fname, s)
    % Helper to save a structure inside parfor
    save(fname, '-struct', 's');
end