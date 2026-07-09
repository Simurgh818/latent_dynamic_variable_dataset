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
nDatasets  = 1; % 10 datasets
k_range    = 10:10; % 10 k components
nK         = numel(k_range);

% Store results: structure indexed by method name
methods = { 'AE'};  % 'PCA', 'AE','ICA'
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
        
        % --- 2. Load Single Dataset ---
        loader = load(fullfile(input_dir, [eegFilename '.mat']));
        s_eeg_all   = double(loader.sim_eeg_vals);
        h_f_all     = double(loader.all_h_F');
        f_peak      = loader.param.f_peak;
        
        % Recalculate parameters locally
        local_param = loader.param; 
        fs         = 1 / loader.dt;
        local_param.fs = fs;
        
        % Determine results directory for this dataset
        subfolderName = ['results_' eegFilename];
        local_results_dir = fullfile(baseFolder, subfolderName);
        if ~exist(local_results_dir, 'dir')
            mkdir(local_results_dir);
        end
        
        % --- 3. Split 80:20 (Train/Test) ---
        idx_split = floor(0.8 * size(s_eeg_all, 2));
        
        eeg_train = s_eeg_all(:, 1:idx_split);
        eeg_test  = s_eeg_all(:, idx_split+1:end);
        
        h_f_train = h_f_all(1:idx_split, :);
        h_f_test  = h_f_all(idx_split+1:end, :);
        
        % --- 4. Strict Z-Score Normalization ---
        % zscore automatically subtracts the mean AND divides by std
        H_train = zscore(h_f_train, 0, 1);
        H_test  = zscore(h_f_test, 0, 1);
        
        data.eeg_train = eeg_train;
        data.eeg_test  = eeg_test;
        data.H_train   = H_train;
        data.H_test    = H_test;
        data.eeg       = s_eeg_all; % For backward compatibility in runDimRedMethod
        data.H_ds      = zscore(h_f_all, 0, 1); % For backward compatibility
        data.f_peak    = f_peak;
        
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
        
        snippets.eeg_train = data.eeg_train; % Can be large, consider snipping if memory is tight
        snippets.param = local_param;
        
        % Collect the reconstruction from each method at the max k
        for m = 1:numel(methods)
            method = methods{m};
            % h_recon_test is already in dataset_res from the runMethod function
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
% 4. Plot Mean Latent Correlation vs # components (k)
% ----------------------------------------------------------
fig1 = figure('Position', [100, 100, 1000, 600]);
for c = 1:numel(conditions)
    cond = conditions{c};
    hold on;
    colors = lines(numel(methods));
    
   for m = 1:numel(methods)
        method = methods{m};
        
        mu_k = mean(STATS.(cond).(method).direct_Component_Corr.mean, 1, 'omitnan');
        sd_k = std(STATS.(cond).(method).direct_Component_Corr.mean, 0, 1, 'omitnan');
        
        errorbar(k_range, mu_k, sd_k, ...
            'LineStyle', method_lines{m}, ...
            'Marker', method_markers{m}, ...
            'LineWidth', 2, ...
            'MarkerSize', 8, ...  
            'Color', colors(m,:), ...
            'DisplayName', method);
    end
    
    xlabel('Number of Components (k)');
    xticks(linspace(1, max(k_range), nK));
    ylabel('Mean correlation between $Z$ and $\hat{Z}$', 'Interpreter', 'latex');
    ylim([0 1]);
    title(sprintf('Mean Latent Correlation vs k'));
    grid on;
    legend('Location', 'eastoutside');
    set(gca, 'FontSize', 20);
end
set(findall(gcf,'-property','FontSize'),'FontSize',20)
summary_trace_name = fullfile(local_results_dir, 'Main_Summary_Corr_vs_k.png');
saveas(fig1, summary_trace_name);

% ==========================================================
% FIGURE 1b: TIME-DOMAIN MATCHED R^2
% ==========================================================
fig1b = figure('Position', [125, 125, 1000, 600]);
for c = 1:numel(conditions)
    cond = conditions{c};
    hold on;
    colors = lines(numel(methods));
    
   for m = 1:numel(methods)
        method = methods{m};
        
        % Average the True Time-Domain R^2 across all latents 
        mu_k_r2 = mean(STATS.(cond).(method).matched_R2.mean, 1, 'omitnan');
        sd_k_r2 = std(STATS.(cond).(method).matched_R2.mean, 0, 1, 'omitnan');
        
        errorbar(k_range, mu_k_r2, sd_k_r2, ...
            'LineStyle', method_lines{m}, ...
            'Marker', method_markers{m}, ...
            'LineWidth', 2, ...
            'MarkerSize', 8, ...  
            'Color', colors(m,:), ...
            'DisplayName', method);
    end
    
    xlabel('Number of Components (k)');
    xticks(k_range); 
    ylabel('Mean R^2');
    
    % If models perform worse than predicting the mean, R^2 drops below 0.
    % You can change the lower limit of ylim here if the lines clip at the bottom.
    ylim([0 1]); 
    title(sprintf('Time-Domain R^2 vs k'));
    grid on;
    legend('Location', 'eastoutside');
    set(gca, 'FontSize', 20);
end
set(findall(fig1b,'-property','FontSize'),'FontSize',20);
summary_r2_name = fullfile(local_results_dir, 'Main_Summary_TimeDomain_R2_vs_k.png');
saveas(fig1b, summary_r2_name);


% ==========================================================
% FIGURE 1c: MATCHED CORRELATION
% ==========================================================
fig1c = figure('Position', [150, 150, 1000, 600]);
for c = 1:numel(conditions)
    cond = conditions{c};
    hold on;
    colors = lines(numel(methods));
    
   for m = 1:numel(methods)
        method = methods{m};
        
        % Preallocate for the k-sweep
        mu_k_matched = zeros(1, nK);
        sd_k_matched = zeros(1, nK);
        
        % Loop through k indices to extract the matched means from CORR_STATS
        for ki = 1:nK
            % Average across all latent variables for this specific k
            mu_k_matched(ki) = mean(CORR_STATS.(cond).(method)(ki).mean, 'omitnan');
            sd_k_matched(ki) = std(CORR_STATS.(cond).(method)(ki).mean, 0, 'omitnan');
        end
        
        errorbar(k_range, mu_k_matched, sd_k_matched, ...
            'LineStyle', method_lines{m}, ...
            'Marker', method_markers{m}, ...
            'LineWidth', 2, ...
            'MarkerSize', 8, ...  
            'Color', colors(m,:), ...
            'DisplayName', method);
    end
    
    xlabel('Number of Components (k)');
    xticks(k_range); 
    ylabel('Mean Matched component to $Z$ correlation', 'Interpreter', 'latex');
    ylim([0 1]);
    title(sprintf('Mean Matched Latent Correlation vs k'));
    grid on;
    legend('Location', 'eastoutside');
    set(gca, 'FontSize', 20);
end
set(findall(fig1c,'-property','FontSize'),'FontSize',20)
summary_matched_name = fullfile(local_results_dir, 'Main_Summary_MatchedCorr_vs_k.png');
saveas(fig1c, summary_matched_name);
%% ----------------------------------------------------------
% 5. Plot Corr and Spectral R^2 vs Peak Frequency (for k=6)
% ----------------------------------------------------------
target_k = 10;
ki_target = find(k_range == target_k);

if ~isempty(ki_target)
    fig2 = figure('Position', [50 50 1700 600]);
    tiledlayout(1, 2, 'Padding', 'compact');
    
    for c = 1:numel(conditions)
        cond = conditions{c};
        
        % --- Left Subplot: Corr (UPDATED TO OPTIMAL MATCHING) ---
        nexttile; hold on;
        colors = lines(numel(methods));
        
        for m = 1:numel(methods)
            method = methods{m};
            
            % Pull from optimally matched CORR_STATS instead of direct STATS
            mu = CORR_STATS.(cond).(method)(ki_target).mean;
            sd = CORR_STATS.(cond).(method)(ki_target).std;
            
            x_col = reshape(double(param.f_peak), [], 1);
            y_col = reshape(double(mu), [], 1);
            e_col = reshape(double(sd), [], 1);
            
            h = errorbar(x_col, y_col, e_col, ...
                'LineStyle', method_lines{m}, ...
                'Marker', method_markers{m}, ...
                'LineWidth', 2, ...
                'MarkerSize', 8, ...
                'Color', colors(m,:));
            
            if numel(h) >= 1
                h(1).DisplayName = char(method);
                for idx = 2:numel(h)
                    h(idx).HandleVisibility = 'off';
                end
            end
        end
        
        xlabel('Peak Frequency (Hz)');
        ylabel('Matched Correlation');
        xticks(param.f_peak);
        ylim([0 1]); 
        title(sprintf('Matched Correlation vs Frequency (k=%d)', target_k));
        grid on; 
        legend('Location','eastoutside'); 
        set(gca, 'FontSize', 20);
        
        % --- Right Subplot: Spectral R^2 ---
        nexttile; hold on;
        
        for m = 1:numel(methods)
            method = methods{m};
            
            mu = STATS.(cond).(method).spectral_R2.mean;
            sd = STATS.(cond).(method).spectral_R2.std;
            
            if size(mu, 2) > 1
                mu = mu(:, ki_target);
                sd = sd(:, ki_target);
            end
            
            x_col = reshape(double(param.f_peak), [], 1);
            y_col = reshape(double(mu), [], 1);
            e_col = reshape(double(sd), [], 1);
            
            h = errorbar(x_col, y_col, e_col, ...
                'LineStyle', method_lines{m}, ...
                'Marker', method_markers{m}, ...
                'LineWidth', 2, ...
                'MarkerSize', 8, ...
                'Color', colors(m,:));
            
            if numel(h) >= 1
                h(1).DisplayName = char(method);
                for idx = 2:numel(h)
                    h(idx).HandleVisibility = 'off';
                end
            end
        end
        
        xlabel('Peak Frequency (Hz)');
        ylabel('Spectral R^2');
        xticks(param.f_peak);
        ylim([0 1]); 
        title(sprintf('Spectral R^2 vs Frequency (k=%d)', target_k));
        grid on; 
        legend('Location','eastoutside'); 
        set(gca, 'FontSize', 20);
    end
    
    set(findall(fig2,'-property','FontSize'),'FontSize',20);
    summary_spectral_name = fullfile(local_results_dir, sprintf('Summary_FreqProfile_k%d.png', target_k));
    saveas(fig2, summary_spectral_name);
else
    % Added an fprintf here to make sure this prints properly in the command window
    fprintf('k=%d is not in your current k_range. Skipping the frequency profile plot.\n', target_k);
end

%% ----------------------------------------------------------
% 6. Plot Matching Correlation vs k (Per Latent Variable)
% ----------------------------------------------------------
for c = 1:numel(conditions)
    cond = conditions{c};
    
    fig3 = figure('Position', [100, 100, 1600, 900]);
    nLatents = length(param.f_peak);
    nCols = ceil(sqrt(nLatents));
    nRows = ceil(nLatents / nCols);
    tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');
    sgtitle(sprintf('Matching Correlation vs k per Latent Variable (%s)', cond), ...
        'FontSize', 24, 'FontWeight', 'bold');
    
    colors = lines(numel(methods));
    
    RESULTS.stats_per_latent.(cond) = struct();
    
    for f = 1:nLatents
        nexttile; hold on;
        peak_freq = param.f_peak(f);
        
        for m = 1:numel(methods)
            method = methods{m};
            
            mu_k = zeros(1, nK);
            sd_k = zeros(1, nK);
            for ki = 1:nK
                mu_k(ki) = CORR_STATS.(cond).(method)(ki).mean(f);
                sd_k(ki) = CORR_STATS.(cond).(method)(ki).std(f);
            end
            
            errorbar(k_range, mu_k, sd_k, ...
                'LineStyle', method_lines{m}, ...
                'Marker', method_markers{m}, ...
                'LineWidth', 2.5, ...
                'MarkerSize', 7, ...
                'Color', colors(m,:), ...
                'DisplayName', method);
            
            latent_name = sprintf('Z_%gHz', peak_freq);
            latent_name = strrep(latent_name, '.', '_'); 
            
            RESULTS.stats_per_latent.(cond).(method).(latent_name).k = k_range;
            RESULTS.stats_per_latent.(cond).(method).(latent_name).corr_mean = mu_k;
            RESULTS.stats_per_latent.(cond).(method).(latent_name).corr_std = sd_k;
        end
        
        xlabel('Number of Components (k)');
        ylabel('Matching Correlation (\rho)');
        xticks(linspace(min(k_range), max(k_range), nK));
        xlim([1, max(k_range)]);
        ylim([0 1]);
        title(sprintf('Latent: %g Hz', peak_freq));
        grid on;
        
        if f == 1
            legend('Location', 'best');
        end
        set(gca, 'FontSize', 16);
    end
    
    fig_name = fullfile(local_results_dir, sprintf('Matching_Corr_vs_k_Per_Latent_%s.png', cond));
    saveas(fig3, fig_name);
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