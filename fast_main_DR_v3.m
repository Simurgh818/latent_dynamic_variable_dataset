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
    'Article Paper' filesep 'simEEG']; 
    baseFolder = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Article Paper' filesep 'Results'];
elseif exist('G:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Article Paper' filesep 'simEEG'];
    baseFolder = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Article Paper' filesep 'Results'];
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
nDatasets  = 1; 

% --- TARGETED RUN PARETERS ---
k_range    = 6; % Only run k=6
nK         = numel(k_range);
methods    = {'PCA', 'AE'}; % Only run PCA and AE
durations  = [10, 60, 360, 8640]; % Data lengths in seconds
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
    parfor dur_idx = 1:nDurations
        dur = durations(dur_idx);
        fprintf('\n   --- Processing Duration: %d seconds ---\n', dur);
        
        % Preallocate temporary storage for parallel workers
        dataset_results = cell(1, nDatasets);
        
        % ---------------------------------------------------------------------
        % PARALLEL LOOP (Across datasets for the CURRENT duration)
        % ---------------------------------------------------------------------
        for d = 1:nDatasets
            fprintf('      Dataset %d / %d (Worker Processing)\n', d, nDatasets);
            data = struct();
            
            % --- 1. Load Data (Local to Worker) ---
            if d < 10 && ~strcmp(cond, 'ou') && ~strcmp(cond,'set4')
                eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, dur);
            elseif d < 10
                eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, dur);
            elseif d == 1 && strcmp(cond, 'ou')
                eegFilename = sprintf('simEEG_Morrell_%s', cond);
            else
                eegFilename = sprintf('simEEG_%s_spat%d_dur%d', cond, d, dur);
            end
            dataset_name = eegFilename;
            
            % Load file
            loader = load(fullfile(input_dir, [eegFilename '.mat']));
            
            % Extract local variables
            s_eeg_like      = double(loader.train_sim_eeg_vals);
            h_f             = double(loader.train_true_hF');
            f_peak          = loader.param.f_peak;
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
            
            % Split Train/Test
            eeg = s_eeg_like; 
            idx_split = floor(0.8 * size(eeg, 2));
            eeg_train = eeg(:, 1:idx_split);
            eeg_test  = eeg(:, idx_split+1:end);
            
            h_f_norm_orig = h_f ./ std(h_f, 0, 1);
            H_train = h_f_norm_orig(1:idx_split, :);
            H_test  = h_f_norm_orig(idx_split+1:end, :);
            
            data.eeg_train = eeg_train;
            data.eeg_test  = eeg_test;
            data.H_train   = H_train;
            data.H_test    = H_test;
            data.eeg       = s_eeg_like; 
            data.H_ds      = h_f_norm_orig;
            data.f_peak    = f_peak;
            
            % --- 3. Method Loop ---
            dataset_res = struct();
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
                
                for ki = 1:nK
                    k = k_range(ki);
                    fprintf('         -> Running %s with k = %d ...\n', method, k);
                    
                    entry = runDimRedMethod( ...
                        method, data, local_param, k, ki, cond, dataset_name, method_dir,...
                        local_results_dir);
                    
                    % Store basic stats                           
                    dataset_res.(method).Comp_latent_matching_corr{ki} = entry.Comp_latent_matching_corr;
                    dataset_res.(method).Comp_latent_matching_matrix{ki} = entry.Comp_latent_matching_matrix;
                    dataset_res.(method).direct_Component_Corr(:,ki) = entry.direct_Component_Corr;
                    dataset_res.(method).spectral_R2(:,ki) = entry.spectral_R2;
                    
                    if ki == nK
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
                        if ~ismember('duration', current_vars), current_corr_table.duration = repmat(dur, nRows, 1); end % Track duration in tables
                        all_d_entries = [all_d_entries; current_corr_table];
                    end
                end
            end
            
            % Pack variables to save cleanly
            ds_out = struct();
            ds_out.analysis = dataset_res;
            ds_out.entries  = all_d_entries;
            ds_out.param    = local_param;
            ds_out.dataset  = d;
            ds_out.cond     = cond;
            ds_out.duration = dur;
            
            ds_filename = fullfile(local_results_dir, sprintf('Results_%s.mat', dataset_name));
            parsave_struct(ds_filename, ds_out);
            
            % Save results for this dataset
            dataset_results{d}.(cond).output_dir = local_results_dir;
            dataset_results{d}.(cond).analysis = dataset_res;
            dataset_results{d}.(cond).entries = all_d_entries;
        end 
        duration_results{dur_idx} = dataset_results;

    end % End Parfor Loop
    
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
    end % End Durations Loop
end % End Conditions Loop

%% ---------------------------------------------------------------------
% STATS AGGREGATION FOR DURATION PLOTS
% ---------------------------------------------------------------------
STATS = struct();
for c = 1:numel(conditions)
    cond = conditions{c};
    for m = 1:numel(methods)
        method = methods{m};
        
        mean_corr = nan(1, nDurations);
        std_corr  = nan(1, nDurations);
        mean_r2   = nan(1, nDurations);
        std_r2    = nan(1, nDurations);
        
        for dur_idx = 1:nDurations
            corr_all = nan(nDatasets, param.N_F);
            r2_all   = nan(nDatasets, param.N_F);
            
            for d = 1:nDatasets
                analysis = EXP.(cond).duration(dur_idx).dataset(d).analysis;
                % Pull the 1st column (since we only ran ki=1 for k=6)
                corr_all(d, :) = analysis.(method).direct_Component_Corr(:, 1);
                r2_all(d, :)   = analysis.(method).spectral_R2(:, 1);
            end
            
            % Calculate mean and standard deviation across ALL latents AND datasets for this duration
            mean_corr(dur_idx) = mean(corr_all(:), 'omitnan');
            std_corr(dur_idx)  = std(corr_all(:), 'omitnan');
            
            mean_r2(dur_idx)   = mean(r2_all(:), 'omitnan');
            std_r2(dur_idx)    = std(r2_all(:), 'omitnan');
        end
        
        STATS.(cond).(method).corr_mean = mean_corr;
        STATS.(cond).(method).corr_std  = std_corr;
        STATS.(cond).(method).r2_mean   = mean_r2;
        STATS.(cond).(method).r2_std    = std_r2;
    end
end

% Save comprehensive outputs
save(fullfile(baseFolder, "RESULTS_DataLength_Benchmark.mat"), "EXP", "STATS", "-v7.3");

%% ----------------------------------------------------------
% FINAL PLOTS: Performance vs. Data Length
% ----------------------------------------------------------
fig1 = figure('Position', [100, 100, 1400, 600]);
tiledlayout(1, 2, 'Padding', 'compact');
colors = lines(numel(methods));
main_title = sprintf('Performance vs. Data Length (k=%d, %s)', k_range(1), conditions{1});
sgtitle(main_title, 'FontSize', 24, 'FontWeight', 'bold');

% --- Subplot 1: Data Length vs Correlation ---
nexttile; hold on;
for m = 1:numel(methods)
    method = methods{m};
    
    mu_dur = STATS.(conditions{1}).(method).corr_mean;
    sd_dur = STATS.(conditions{1}).(method).corr_std;
    
    errorbar(durations, mu_dur, sd_dur, '-o', ...
        'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(m,:), 'DisplayName', method);
end
set(gca, 'XScale', 'log'); % Log scale to handle 10 vs 8640 cleanly
xticks(durations);
xticklabels(string(durations));
xlabel('Data Length (seconds)');
ylabel('Mean Correlation (\rho)');
ylim([0 1]);
title('Latent Component Correlation');
grid on; legend('Location', 'best');
set(gca, 'FontSize', 18);

% --- Subplot 2: Data Length vs Spectral R^2 ---
nexttile; hold on;
for m = 1:numel(methods)
    method = methods{m};
    
    mu_r2 = STATS.(conditions{1}).(method).r2_mean;
    sd_r2 = STATS.(conditions{1}).(method).r2_std;
    
    errorbar(durations, mu_r2, sd_r2, '-o', ...
        'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(m,:), 'DisplayName', method);
end
set(gca, 'XScale', 'log');
xticks(durations);
xticklabels(string(durations));
xlabel('Data Length (seconds)');
ylabel('Mean Spectral R^2');
ylim([0 1]);
title('Spectral R^2');
grid on; legend('Location', 'best');
set(gca, 'FontSize', 18);

% Save Figure
summary_duration_name = fullfile(baseFolder, sprintf('DataLength_vs_Performance_k%d.png', k_range(1)));
saveas(fig1, summary_duration_name);

%% ---------------------------------------------------------------------
%  HELPER FUNCTIONS
% ---------------------------------------------------------------------
function parsave_struct(fname, s)
    % Helper to save a structure inside parfor
    save(fname, '-struct', 's');
end