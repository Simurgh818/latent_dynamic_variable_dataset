%% MAIN SCRIPT FOR DIMENSIONALITY REDUCTION BENCHMARK: DATA LENGTH ANALYSIS
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
    input_dir = [path_to_files filesep 'Shared Code' filesep 'simEEG'];
    baseFolder = [path_to_files filesep 'Dimensionality Reduction Review Paper'];    
else
    error('Unknown system: Cannot determine input and output paths.');
end

%% Loop through experiments
conditions = {'set4'}; 
nDatasets  = 10; 
% --- TARGETED RUN PARAMETERS ---
k_range    = 10; 
nK         = numel(k_range);
methods    = {'PCA', 'AE', 'ICA'}; 
durations  = [1, 5, 10, 60, 1000, 3600, 10800]; % Data lengths in seconds 
nDurations = numel(durations);
% -----------------------------
EXP = struct();
param = struct();
RESULTS = struct();
RESULTS.meta = struct();
RESULTS.meta.created = datetime;
RESULTS.meta.description = "Data Length vs Performance benchmark";

target_workers = 5; 
current_pool = gcp('nocreate');
if isempty(current_pool)
    parpool(target_workers);
elseif current_pool.NumWorkers ~= target_workers
    delete(current_pool);
    parpool(target_workers);
end

%% Loop through conditions and durations
for c = 1:numel(conditions)
    cond = conditions{c};
    fprintf('\n=== Running condition: %s ===\n', cond);
    duration_results = cell(1, nDurations);
    
    % SERIAL LOOP OVER DURATIONS (Prevents multiple workers from running 3600s + 10800s simultaneously)
    for dur_idx = 1:nDurations
        dur = durations(dur_idx);
        fprintf('\n   --- Processing Duration: %d seconds ---\n', dur);
        
        % Dynamically reduce worker count for extreme durations to save RAM
        if dur >= 3600
            max_active_workers = 2; % Cap at 2 workers for 1h and 3h datasets
        else
            max_active_workers = target_workers;
        end
        
        % Preallocate temporary storage for parallel workers
        dataset_results = cell(1, nDatasets);
        
        % ---------------------------------------------------------------------
        % PARALLEL LOOP (Across datasets for the CURRENT duration)
        % ---------------------------------------------------------------------
        parfor (d = 1:nDatasets, max_active_workers)
            fprintf('      Dataset %d / %d (Duration: %ds)\n', d, nDatasets, dur);
            
            % --- 1. Determine Filenames (Local to Worker) ---
            if d < 10 
                eegFilename  = sprintf('simEEG_set4_spat0%d_dur%d', d, dur);
            else
                eegFilename  = sprintf('simEEG_set4_spat%d_dur%d', d, dur);
            end
            testFilename = 'simEEG_set4_spat01_dur1728';
            dataset_name = eegFilename;
            
            % Determine results directory for this dataset
            subfolderName = ['results_' eegFilename];
            local_results_dir = fullfile(baseFolder, subfolderName);
            if ~exist(local_results_dir, 'dir')
                mkdir(local_results_dir);
            end
            ds_filename = fullfile(local_results_dir, sprintf('Results_%s.mat', dataset_name));
            
            % --- SMART CHECKPOINT RESUME: Validate existing file before skipping ---
            if exist(ds_filename, 'file')
                loaded_ds = load(ds_filename);
                is_complete = isfield(loaded_ds, 'analysis') && isfield(loaded_ds, 'entries');
                
                % Verify all requested methods exist and completed (not empty)
                if is_complete
                    for m_chk = 1:numel(methods)
                        m_name = methods{m_chk};
                        if ~isfield(loaded_ds.analysis, m_name) || ...
                           isempty(loaded_ds.analysis.(m_name).Comp_latent_matching_corr{1})
                            is_complete = false;
                            break;
                        end
                    end
                end
                
                if is_complete
                    % Backfill dim_estimations if this file came from an older run!
                    if ~isfield(loaded_ds.analysis, 'dim_estimations')
                        fprintf('      [Updating] Backfilling dim_estimations for %s\n', dataset_name);
                        tmp_train = load(fullfile(input_dir, [eegFilename '.mat']), 'sim_eeg_vals');
                        eeg_tmp   = double(tmp_train.sim_eeg_vals);
                        tmp_train = [];
                        
                        dim_est = struct();
                        dim_est.rank = rank(eeg_tmp);
                        [p_ch, n_t] = size(eeg_tmp);
                        eigVals = sort(eig(cov(eeg_tmp')), 'descend');
                        noise_var = median(eigVals);
                        mp_thresh = noise_var * (1 + sqrt(p_ch/n_t))^2;
                        dim_est.mp = sum(eigVals > mp_thresh);
                        try
                            [dim_est.map, ~] = velicer_map(eeg_tmp', false);
                        catch
                            dim_est.map = NaN;
                        end
                        eeg_tmp = [];
                        
                        loaded_ds.analysis.dim_estimations = dim_est;
                        parsave_struct(ds_filename, loaded_ds);
                    else
                        fprintf('      [Skipping] Found complete results for %s\n', dataset_name);
                    end
                    
                    dataset_results{d}.(cond).output_dir = local_results_dir;
                    dataset_results{d}.(cond).analysis   = loaded_ds.analysis;
                    dataset_results{d}.(cond).entries    = loaded_ds.entries;
                    continue;
                else
                    fprintf('      [Re-running] Incomplete/older checkpoint found for %s\n', dataset_name);
                end
            end
            
            % --- 2. Load Train & Test Data (Eliminate redundant copies) ---
            loader_train   = load(fullfile(input_dir, [eegFilename '.mat']));
            local_param    = loader_train.param; 
            local_param.fs = 1 / loader_train.dt;
            f_peak         = local_param.f_peak;
            
            data = struct();
            data.eeg_train = double(loader_train.sim_eeg_vals);
            h_f_train      = double(loader_train.all_h_F');
            loader_train   = []; % Immediately free raw struct from worker RAM
            
            loader_test    = load(fullfile(input_dir, [testFilename '.mat']));
            data.eeg_test  = double(loader_test.sim_eeg_vals);
            h_f_test       = double(loader_test.all_h_F');
            loader_test    = []; % Immediately free raw struct from worker RAM
            
            % Normalize latents individually to zero-mean, unit-variance
            data.H_train   = h_f_train ./ std(h_f_train, 0, 1);
            data.H_test    = h_f_test  ./ std(h_f_test, 0, 1);
            h_f_train = []; h_f_test = []; % Clear unnormalized copies
            
            % Backward compatibility copies
            data.eeg       = data.eeg_train; 
            data.H_ds      = data.H_train;     
            data.f_peak    = f_peak;
            
            % =================================================================
            % --- Intrinsic Dimensionality Estimations ---
            % =================================================================
            dim_estimations = struct();
            dim_estimations.rank = rank(data.eeg_train);
            
            [p_ch, n_t] = size(data.eeg_train);
            eigVals = sort(eig(cov(data.eeg_train')), 'descend');
            noise_var = median(eigVals); 
            mp_thresh = noise_var * (1 + sqrt(p_ch/n_t))^2;
            dim_estimations.mp = sum(eigVals > mp_thresh);
            
            try
                % Pass false to suppress figure creation inside parfor!
                [dim_estimations.map, ~] = velicer_map(data.eeg_train', false); 
            catch
                warning('velicer_map failed. Storing NaN.');
                dim_estimations.map = NaN;
            end
            
            % --- Method Loop ---
            dataset_res = struct();
            dataset_res.dim_estimations = dim_estimations;
            dataset_res.f_peak = local_param.f_peak;
            all_d_entries = table();
            
            for m = 1:numel(methods)
                method = methods{m};
                method_dir = fullfile(local_results_dir, method);
                if ~exist(method_dir, 'dir'), mkdir(method_dir); end
                
                dataset_res.(method).Comp_latent_matching_corr = cell(1, nK);
                dataset_res.(method).Comp_latent_matching_matrix = cell(1, nK);
                dataset_res.(method).direct_Component_Corr = nan(local_param.N_F, nK);
                dataset_res.(method).spectral_R2 = nan(local_param.N_F, nK);
                dataset_res.(method).h_recon_test = [];
                
                for ki = 1:nK
                    k = k_range(ki);
                    fprintf('         -> Running %s with k = %d (Dur: %ds, DS: %d)...\n', method, k, dur, d);
                    
                    % --- FAIL-SAFE TRY-CATCH BLOCK ---
                    try
                        entry = runDimRedMethod( ...
                            method, data, local_param, k, ki, cond, dataset_name, method_dir, ...
                            local_results_dir);
                        
                        % Store basic stats                           
                        dataset_res.(method).Comp_latent_matching_corr{ki}   = entry.Comp_latent_matching_corr;
                        dataset_res.(method).Comp_latent_matching_matrix{ki} = entry.Comp_latent_matching_matrix;
                        dataset_res.(method).direct_Component_Corr(:,ki)     = entry.direct_Component_Corr;
                        dataset_res.(method).spectral_R2(:,ki)               = entry.spectral_R2;
                        
                        if ki == nK && isfield(entry, 'out') && isfield(entry.out, 'h_recon_test')
                            dataset_res.(method).h_recon_test = entry.out.h_recon_test;
                        end
                        
                        current_corr_table = entry.Comp_latent_matching_corr; 
                        if ~isempty(current_corr_table)
                            nRows = height(current_corr_table);
                            current_vars = current_corr_table.Properties.VariableNames;
                            if ~ismember('method', current_vars), current_corr_table.method = repmat(categorical(cellstr(method)), nRows, 1); end
                            if ~ismember('dataset', current_vars), current_corr_table.dataset = repmat(d, nRows, 1); end
                            if ~ismember('condition', current_vars), current_corr_table.condition = repmat(categorical(cellstr(cond)), nRows, 1); end
                            if ~ismember('k', current_vars), current_corr_table.k = repmat(k, nRows, 1); end
                            if ~ismember('duration', current_vars), current_corr_table.duration = repmat(dur, nRows, 1); end
                            all_d_entries = [all_d_entries; current_corr_table];
                        end
                        
                    catch ME
                        warning('FAILED on Method: %s | Dur: %ds | Dataset: %d | k: %d\nReason: %s', ...
                            method, dur, d, k, ME.message);
                        
                        if gpuDeviceCount > 0
                            try gpuDevice([]); catch, end
                        end
                    end
                    % ---------------------------------
                end
            end
            
            % Clear heavy data struct before saving
            data = [];
            
            % Pack variables to save cleanly
            ds_out = struct();
            ds_out.analysis = dataset_res;
            ds_out.entries  = all_d_entries;
            ds_out.param    = local_param;
            ds_out.dataset  = d;
            ds_out.cond     = cond;
            ds_out.duration = dur;
            
            parsave_struct(ds_filename, ds_out);
            
            % Save results for this dataset
            dataset_results{d}.(cond).output_dir = local_results_dir;
            dataset_results{d}.(cond).analysis   = dataset_res;
            dataset_results{d}.(cond).entries    = all_d_entries;
        end % End Parfor Loop
        
        duration_results{dur_idx} = dataset_results;
    end % End Durations Loop
    
    % --- NOW SAFELY BUILD THE EXP STRUCT (Serial) ---
    for dur_idx = 1:nDurations
        dataset_results = duration_results{dur_idx};
        
        for d = 1:nDatasets
            EXP.(cond).duration(dur_idx).dataset(d) = dataset_results{d}.(cond);
            
            % Update param only once
            if dur_idx == 1 && d == 1
                 param.f_peak = dataset_results{d}.(cond).analysis.f_peak;
                 param.N_F = length(param.f_peak);
            end
        end
    end 
end % End Conditions Loop

%% ---------------------------------------------------------------------
% STATS AGGREGATION FOR DURATION PLOTS
% ---------------------------------------------------------------------
STATS = struct();
for c = 1:numel(conditions)
    cond = conditions{c};
    STATS.(cond).dim_estimations = struct();
    
    for dur_idx = 1:nDurations
        rank_all = nan(nDatasets, 1);
        mp_all   = nan(nDatasets, 1);
        map_all  = nan(nDatasets, 1);
        
        for d = 1:nDatasets
            analysis = EXP.(cond).duration(dur_idx).dataset(d).analysis;
            if isfield(analysis, 'dim_estimations')
                rank_all(d) = analysis.dim_estimations.rank;
                mp_all(d)   = analysis.dim_estimations.mp;
                map_all(d)  = analysis.dim_estimations.map;
            end
        end
        
        STATS.(cond).dim_estimations.rank_mean(dur_idx) = mean(rank_all, 'omitnan');
        STATS.(cond).dim_estimations.rank_std(dur_idx)  = std(rank_all, 'omitnan');
        STATS.(cond).dim_estimations.mp_mean(dur_idx)   = mean(mp_all, 'omitnan');
        STATS.(cond).dim_estimations.mp_std(dur_idx)    = std(mp_all, 'omitnan');
        STATS.(cond).dim_estimations.map_mean(dur_idx)  = mean(map_all, 'omitnan');
        STATS.(cond).dim_estimations.map_std(dur_idx)   = std(map_all, 'omitnan');
    end
end

for c = 1:numel(conditions)
    cond = conditions{c};
    for m = 1:numel(methods)
        method = methods{m};
        
        mean_matching_corr = nan(1, nDurations);
        std_matching_corr  = nan(1, nDurations);
        mean_r2            = nan(1, nDurations);
        std_r2             = nan(1, nDurations);
        
        matching_corr_per_latent_mean = nan(nDurations, param.N_F);
        matching_corr_per_latent_std  = nan(nDurations, param.N_F);
        r2_per_latent_mean            = nan(nDurations, param.N_F);
        r2_per_latent_std             = nan(nDurations, param.N_F);
        
        for dur_idx = 1:nDurations
            matching_corr_all = nan(nDatasets, param.N_F);
            r2_all            = nan(nDatasets, param.N_F);
            
            for d = 1:nDatasets
                analysis = EXP.(cond).duration(dur_idx).dataset(d).analysis;
                match_vals = nan(1, param.N_F);
                
                if isfield(analysis, method)
                    match_table = analysis.(method).Comp_latent_matching_corr{1};
                    if istable(match_table) && ~isempty(match_table)
                        colNames = match_table.Properties.VariableNames;
                        if ismember('corr_value', colNames)
                            match_vals = match_table.corr_value(:)';
                        end
                    end
                    r2_all(d, :) = analysis.(method).spectral_R2(:, 1);
                end
                
                matching_corr_all(d, :) = match_vals;
            end
            
            mean_matching_corr(dur_idx) = mean(matching_corr_all(:), 'omitnan');
            std_matching_corr(dur_idx)  = std(matching_corr_all(:), 'omitnan');
            mean_r2(dur_idx)            = mean(r2_all(:), 'omitnan');
            std_r2(dur_idx)             = std(r2_all(:), 'omitnan');
            
            matching_corr_per_latent_mean(dur_idx, :) = mean(matching_corr_all, 1, 'omitnan');
            matching_corr_per_latent_std(dur_idx, :)  = std(matching_corr_all, 0, 1, 'omitnan');
            r2_per_latent_mean(dur_idx, :)            = mean(r2_all, 1, 'omitnan');
            r2_per_latent_std(dur_idx, :)             = std(r2_all, 0, 1, 'omitnan');
        end
        
        STATS.(cond).(method).matching_corr_mean = mean_matching_corr;
        STATS.(cond).(method).matching_corr_std  = std_matching_corr;
        STATS.(cond).(method).r2_mean            = mean_r2;
        STATS.(cond).(method).r2_std             = std_r2;
        
        STATS.(cond).(method).matching_corr_per_latent_mean = matching_corr_per_latent_mean;
        STATS.(cond).(method).matching_corr_per_latent_std  = matching_corr_per_latent_std;
        STATS.(cond).(method).r2_per_latent_mean            = r2_per_latent_mean;
        STATS.(cond).(method).r2_per_latent_std             = r2_per_latent_std;
    end
end

%% ---------------------------------------------------------------------
% SAVE RESULTS STRUCTURE
% ---------------------------------------------------------------------
RESULTS.PerLatent = struct();
for m = 1:numel(methods)
    method = methods{m};
    RESULTS.PerLatent.(method).duration = durations;
    RESULTS.PerLatent.(method).f_peak = param.f_peak;
    RESULTS.PerLatent.(method).matching_corr_mean = STATS.(conditions{1}).(method).matching_corr_per_latent_mean;
    RESULTS.PerLatent.(method).matching_corr_std = STATS.(conditions{1}).(method).matching_corr_per_latent_std;
end

RESULTS.IntrinsicDim = struct();
RESULTS.IntrinsicDim.duration  = durations;
RESULTS.IntrinsicDim.rank_mean = STATS.(conditions{1}).dim_estimations.rank_mean;
RESULTS.IntrinsicDim.rank_std  = STATS.(conditions{1}).dim_estimations.rank_std;
RESULTS.IntrinsicDim.mp_mean   = STATS.(conditions{1}).dim_estimations.mp_mean;
RESULTS.IntrinsicDim.mp_std    = STATS.(conditions{1}).dim_estimations.mp_std;
RESULTS.IntrinsicDim.map_mean  = STATS.(conditions{1}).dim_estimations.map_mean;
RESULTS.IntrinsicDim.map_std   = STATS.(conditions{1}).dim_estimations.map_std;

save(fullfile(baseFolder, "RESULTS_DataLength_Benchmark.mat"), "EXP", "STATS", "RESULTS", "-v7.3");

%% ----------------------------------------------------------
% FINAL PLOTS (Fixed: Removed unrecognized 'MarkerFaceAlpha' from errorbar)
% ----------------------------------------------------------
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
colors = cb_palette(mod(0:numel(methods)-1, size(cb_palette,1)) + 1, :);
method_markers = {'o', 's', '^', 'd', 'v', 'p'};
method_lines   = {'-', '--', '-.', ':', '-', '--'};

% Fixed labels to match all 7 elements of durations: [1, 5, 10, 60, 1000, 3600, 10800]
lbls = {'1s', '5s', '10s', '1m', '16.7m', '1h', '3h'};

% === FIGURE 1: Global Performance vs. Data Length ===
fig1 = figure('Position', [100, 100, 1700, 600], 'Visible', 'off');
tiledlayout(1, 2, 'Padding', 'compact');
sgtitle(sprintf('Performance vs. Data Length (k=%d)', k_range(1)), 'FontSize', 28, 'FontWeight', 'bold');

nexttile; hold on;
set(gca, 'XScale', 'log', 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20, 'FontWeight', 'bold');
for m = 1:numel(methods)
    method = methods{m};
    m_idx = mod(m-1, length(method_markers)) + 1;
    l_idx = mod(m-1, length(method_lines)) + 1;
    face_col = 0.6 * colors(m,:) + 0.4 * [1, 1, 1];
    errorbar(durations, STATS.(conditions{1}).(method).matching_corr_mean, STATS.(conditions{1}).(method).matching_corr_std, ...
        'LineStyle', method_lines{l_idx}, 'Marker', method_markers{m_idx}, ...
        'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(m,:), ...
        'MarkerFaceColor', face_col, 'DisplayName', method);
end
xticks(durations); xticklabels(lbls);
xlabel('Data Length', 'FontSize', 22, 'FontWeight', 'bold');
ylabel('Mean Max Correlation (\rho)', 'FontSize', 22, 'FontWeight', 'bold');
ylim([0 1]); title('Global Latent Matching Correlation', 'FontSize', 24, 'FontWeight', 'bold');
grid on; lgd1 = legend('Location', 'best'); lgd1.FontSize = 18;

nexttile; hold on;
set(gca, 'XScale', 'log', 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20, 'FontWeight', 'bold');
for m = 1:numel(methods)
    method = methods{m};
    m_idx = mod(m-1, length(method_markers)) + 1;
    l_idx = mod(m-1, length(method_lines)) + 1;
    face_col = 0.6 * colors(m,:) + 0.4 * [1, 1, 1];
    errorbar(durations, STATS.(conditions{1}).(method).r2_mean, STATS.(conditions{1}).(method).r2_std, ...
        'LineStyle', method_lines{l_idx}, 'Marker', method_markers{m_idx}, ...
        'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(m,:), ...
        'MarkerFaceColor', face_col, 'DisplayName', method);
end
xticks(durations); xticklabels(lbls);
xlabel('Data Length', 'FontSize', 22, 'FontWeight', 'bold');
ylabel('Mean Spectral R^2', 'FontSize', 22, 'FontWeight', 'bold');
ylim([0 1]); title('Spectral R^2', 'FontSize', 24, 'FontWeight', 'bold');
grid on; lgd2 = legend('Location', 'best'); lgd2.FontSize = 18;

out_fig1 = fullfile(baseFolder, sprintf('DataLength_vs_Performance_k%d.png', k_range(1)));
exportgraphics(fig1, out_fig1, 'Resolution', 600);
% close(fig1);

% === FIGURE 2: Spectral R2 Frequency Profile per Duration ===
if nDurations > 1 && nDurations <= 8
    fig2 = figure('Position', [150, 150, 1600, 950], 'Color', 'w', 'Visible', 'off');
    nCols = ceil(sqrt(nDurations));
    nRows = ceil(nDurations / nCols);
    t2 = tiledlayout(nRows, nCols, 'Padding', 'normal', 'TileSpacing', 'compact');
    
    for dur_idx = 1:nDurations
        nexttile; hold on;
        set(gca, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 18, 'FontWeight', 'bold');
        
        for m = 1:numel(methods)
            method = methods{m};
            m_idx = mod(m-1, length(method_markers)) + 1;
            l_idx = mod(m-1, length(method_lines)) + 1;
            face_col = 0.6 * colors(m,:) + 0.4 * [1, 1, 1];
            
            mu = STATS.(conditions{1}).(method).r2_per_latent_mean(dur_idx, :);
            sd = STATS.(conditions{1}).(method).r2_per_latent_std(dur_idx, :);
            
            errorbar(reshape(double(param.f_peak),[],1), reshape(double(mu),[],1), reshape(double(sd),[],1), ...
                     'LineStyle', method_lines{l_idx}, 'Marker', method_markers{m_idx}, ...
                     'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(m,:), ...
                     'MarkerFaceColor', face_col, 'DisplayName', method);
        end
        
        xlabel('Peak Frequency (Hz)', 'FontSize', 22, 'FontWeight', 'bold'); 
        ylabel('Spectral R^2', 'FontSize', 22, 'FontWeight', 'bold');
        xticks(param.f_peak); 
        ylim([0 1]);
        
        % --- Format Duration Title in sec, min, or hours ---
        dur_sec = durations(dur_idx);
        if dur_sec < 60
            dur_str = sprintf('%d sec', dur_sec);
        elseif dur_sec < 3600
            dur_min = dur_sec / 60;
            if mod(dur_sec, 60) == 0
                dur_str = sprintf('%d min', dur_min);
            else
                dur_str = sprintf('%.1f min', dur_min);
            end
        else
            dur_hr = dur_sec / 3600;
            if dur_hr == 1
                dur_str = '1 hour';
            elseif mod(dur_sec, 3600) == 0
                dur_str = sprintf('%d hours', dur_hr);
            else
                dur_str = sprintf('%.1f hours', dur_hr);
            end
        end
        
        title(sprintf('Duration: %s', dur_str), 'FontSize', 22, 'FontWeight', 'bold');
        grid on; 
        
        if dur_idx == 1
            lgd_f2 = legend('Location', 'best'); 
            lgd_f2.FontSize = 16; 
        end
    end
    
    % Placed after loop using tiledlayout handle so exportgraphics doesn't clip it
    title(t2, 'Spectral R^2 vs. Peak Frequency across Data Lengths', 'FontSize', 28, 'FontWeight', 'bold');
    
    out_fig2 = fullfile(baseFolder, sprintf('FreqProfile_vs_DataLength_k%d.png', k_range(1)));
    exportgraphics(fig2, out_fig2, 'Resolution', 600);
    close(fig2);
end

% === FIGURE 3: Data Length vs. Correlation per Latent Variable ===
fig3 = figure('Position', [200, 200, 1600, 900], 'Visible', 'off');
nCols_lat = ceil(sqrt(param.N_F));
nRows_lat = ceil(param.N_F / nCols_lat);
tiledlayout(nRows_lat, nCols_lat, 'Padding', 'compact', 'TileSpacing', 'compact');
sgtitle('Data Length vs. Matching Correlation per True Latent', 'FontSize', 28, 'FontWeight', 'bold');

for f = 1:param.N_F
    nexttile; hold on;
    set(gca, 'XScale', 'log', 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 18, 'FontWeight', 'bold');
    for m = 1:numel(methods)
        method = methods{m};
        m_idx = mod(m-1, length(method_markers)) + 1;
        l_idx = mod(m-1, length(method_lines)) + 1;
        face_col = 0.6 * colors(m,:) + 0.4 * [1, 1, 1];
        mu_f = STATS.(conditions{1}).(method).matching_corr_per_latent_mean(:, f);
        sd_f = STATS.(conditions{1}).(method).matching_corr_per_latent_std(:, f);
        errorbar(reshape(double(durations),[],1), reshape(double(mu_f),[],1), reshape(double(sd_f),[],1), ...
                 'LineStyle', method_lines{l_idx}, 'Marker', method_markers{m_idx}, ...
                 'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(m,:), ...
                 'MarkerFaceColor', face_col, 'DisplayName', method);
    end
    xticks(durations); xticklabels(lbls);
    xlabel('Data Length', 'FontSize', 22, 'FontWeight', 'bold'); 
    ylabel('Max Correlation (\rho)', 'FontSize', 22, 'FontWeight', 'bold');
    ylim([0 1]); title(sprintf('Latent %d (%.1f Hz)', f, param.f_peak(f)), 'FontSize', 22, 'FontWeight', 'bold');
    grid on; 
    if f == 1, lgd_f3 = legend('Location', 'best'); lgd_f3.FontSize = 16; end
end
out_fig3 = fullfile(baseFolder, sprintf('DataLength_vs_MatchingCorr_PerLatent_k%d.png', k_range(1)));
exportgraphics(fig3, out_fig3, 'Resolution', 600);
% close(fig3);

% === FIGURE 4: Intrinsic Dimensionality vs. Data Length ===
fig4 = figure('Position', [300, 300, 900, 600], 'Visible', 'off');
hold on;
set(gca, 'XScale', 'log', 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20, 'FontWeight', 'bold');
yline(param.N_F, '--k', sprintf('True Latents (N=%d)', param.N_F), ...
    'LineWidth', 2.5, 'LabelHorizontalAlignment', 'left', 'FontSize', 18, 'FontWeight', 'bold', 'HandleVisibility', 'off');
col_rank = cb_palette(8, :); col_mp = cb_palette(1, :); col_map = cb_palette(2, :);
face_rank = 0.6 * col_rank + 0.4 * [1, 1, 1];
face_mp   = 0.6 * col_mp   + 0.4 * [1, 1, 1];
face_map  = 0.6 * col_map  + 0.4 * [1, 1, 1];

errorbar(durations, RESULTS.IntrinsicDim.rank_mean, RESULTS.IntrinsicDim.rank_std, ...
         '-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', face_rank, ...
         'Color', col_rank, 'DisplayName', 'Matrix Rank');
errorbar(durations, RESULTS.IntrinsicDim.mp_mean, RESULTS.IntrinsicDim.mp_std, ...
         '-s', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', face_mp, ...
         'Color', col_mp, 'DisplayName', 'Marchenko-Pastur');
errorbar(durations, RESULTS.IntrinsicDim.map_mean, RESULTS.IntrinsicDim.map_std, ...
         '-^', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', face_map, ...
         'Color', col_map, 'DisplayName', 'Velicer''s MAP');

xticks(durations); xticklabels(lbls);
xlabel('Data Length', 'FontSize', 22, 'FontWeight', 'bold');
ylabel('Estimated Dimensionality', 'FontSize', 22, 'FontWeight', 'bold');
title('Dimensionality Estimation vs. Data Length', 'FontSize', 26, 'FontWeight', 'bold');
max_y = max([RESULTS.IntrinsicDim.rank_mean(:); param.N_F]);
ylim([0 max_y + 2]); grid on;
lgd_f4 = legend('Location', 'best'); lgd_f4.FontSize = 18;
out_fig4 = fullfile(baseFolder, 'IntrinsicDimensionality_vs_DataLength.png');
exportgraphics(fig4, out_fig4, 'Resolution', 600);
% close(fig4);

%% ---------------------------------------------------------------------
%  HELPER FUNCTIONS
% ---------------------------------------------------------------------
function parsave_struct(fname, s)
    save(fname, '-struct', 's');
end

function [num_factors, map_values] = velicer_map(data, do_plot)
    if nargin < 2
        do_plot = false; % Default to false to prevent memory leaks in parfor
    end
    R = corrcoef(data);
    p = size(R, 1);
    [eigvec, eigval_mat] = eig(R);
    eigval = diag(eigval_mat);
    [eigval, idx] = sort(eigval, 'descend');
    eigvec = eigvec(:, idx);
    
    map_values = zeros(1, p);
    R_off_diag = R - eye(p);
    map_values(1) = sum(R_off_diag(:).^2) / (p * (p - 1));
    
    for m = 1:(p-1)
        A = eigvec(:, 1:m) * diag(sqrt(eigval(1:m)));
        part_cov = R - (A * A');
        d = diag(part_cov);
        inv_sqrt_d = diag(1 ./ sqrt(d));
        R_partial = inv_sqrt_d * part_cov * inv_sqrt_d;
        R_partial_off = R_partial - eye(p);
        map_values(m+1) = sum(R_partial_off(:).^2) / (p * (p - 1));
    end
    
    [min_val, min_idx] = min(map_values);
    num_factors = min_idx - 1;
    
    if do_plot
        figure('Position',[50 50 800 500]);
        plot(0:(p-1), map_values, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
        hold on;
        plot(num_factors, map_values(min_idx), 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
        coordinate_text = sprintf('  Min: (%d, %.4f)', num_factors, min_val);
        text(num_factors, 0.2, coordinate_text, 'VerticalAlignment', 'top', ...
            'HorizontalAlignment', 'center', 'FontSize', 26, 'FontWeight', 'bold', 'Color', 'r');
        xlabel('Number of Factors Retained'); ylabel('Avg. Sqrd. Partial Corr.');
        title('Velicer''s MAP Test'); legend('MAP Values', 'Minimum (Optimal Factors)');
        set(gca,'FontSize',24); grid on; hold off;
    end
end