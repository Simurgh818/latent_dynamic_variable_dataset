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
nDatasets  = 5; 

% --- TARGETED RUN PARETERS ---
k_range    = 6; % Only run k=6
nK         = numel(k_range);
methods    = {'PCA'}; % ,'AE','ICA'
durations  = [10, 60, 360, 8640]; %  Data lengths in seconds 
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
    parfor dur_idx = 1:nDurations % parfor
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
            s_eeg_like      = double(loader.sim_eeg_vals);
            h_f             = double(loader.all_h_F');
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

            % =================================================================
            % --- Intrinsic Dimensionality Estimations ---
            % =================================================================
            dim_estimations = struct();
            
            % 1. Rank of the training data
            dim_estimations.rank = rank(eeg_train);
            
            % 2. Marchenko-Pastur Threshold
            % X must be (channels x time). We compute cov of X' (time x channels)
            [p_ch, n_t] = size(eeg_train);
            eigVals = sort(eig(cov(eeg_train')), 'descend');
            noise_var = median(eigVals); % Heuristic estimate of noise variance
            mp_thresh = noise_var * (1 + sqrt(p_ch/n_t))^2;
            dim_estimations.mp = sum(eigVals > mp_thresh);
            
            % 3. Velicer's MAP test
            % Assumes you saved the velicer_map.m script in your path.
            % It expects rows = observations (time), columns = variables (channels).
            try
                [dim_estimations.map, ~] = velicer_map(eeg_train'); 
            catch
                warning('velicer_map.m not found or failed. Storing NaN.');
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
% --- Aggregate Intrinsic Dimensionality Estimates ---
for c = 1:numel(conditions)
    cond = conditions{c};
    STATS.(cond).dim_estimations = struct();
    
    for dur_idx = 1:nDurations
        rank_all = nan(nDatasets, 1);
        mp_all   = nan(nDatasets, 1);
        map_all  = nan(nDatasets, 1);
        
        for d = 1:nDatasets
            analysis = EXP.(cond).duration(dur_idx).dataset(d).analysis;
            rank_all(d) = analysis.dim_estimations.rank;
            mp_all(d)   = analysis.dim_estimations.mp;
            map_all(d)  = analysis.dim_estimations.map;
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
        
        % Global Means (Averaged across all latents)
        mean_matching_corr = nan(1, nDurations);
        std_matching_corr  = nan(1, nDurations);
        mean_r2            = nan(1, nDurations);
        std_r2             = nan(1, nDurations);
        
        % Per-Latent Means (For the new Figure 3)
        matching_corr_per_latent_mean = nan(nDurations, param.N_F);
        matching_corr_per_latent_std  = nan(nDurations, param.N_F);
        r2_per_latent_mean            = nan(nDurations, param.N_F);
        r2_per_latent_std             = nan(nDurations, param.N_F);
        
        for dur_idx = 1:nDurations
            matching_corr_all = nan(nDatasets, param.N_F);
            r2_all            = nan(nDatasets, param.N_F);
            
            for d = 1:nDatasets
                analysis = EXP.(cond).duration(dur_idx).dataset(d).analysis;
                
                % --- PULL FROM MATCHING TABLE (ki=1 for k=6) ---
                match_table = analysis.(method).Comp_latent_matching_corr{1};
                
                % Dynamically find the correlation column name
                colNames = match_table.Properties.VariableNames;
                if ismember('corr_value', colNames)
                    match_vals = match_table.corr_value;
               
                end
                
                % Store values
                matching_corr_all(d, :) = match_vals;
                r2_all(d, :)            = analysis.(method).spectral_R2(:, 1);
            end
            
            % Global metrics (collapsed across latents)
            mean_matching_corr(dur_idx) = mean(matching_corr_all(:), 'omitnan');
            std_matching_corr(dur_idx)  = std(matching_corr_all(:), 'omitnan');
            mean_r2(dur_idx)            = mean(r2_all(:), 'omitnan');
            std_r2(dur_idx)             = std(r2_all(:), 'omitnan');
            
            % Per-latent metrics
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

% Save comprehensive outputs
save(fullfile(baseFolder, "RESULTS_DataLength_Benchmark.mat"), "EXP", "STATS", "RESULTS", "-v7.3");

%% ----------------------------------------------------------
% FINAL PLOTS
% ----------------------------------------------------------
colors = lines(numel(methods));

% === FIGURE 1: Global Performance vs. Data Length ===
fig1 = figure('Position', [100, 100, 1400, 600]);
tiledlayout(1, 2, 'Padding', 'compact');
sgtitle(sprintf('Performance vs. Data Length (k=%d)', k_range(1)), 'FontSize', 24, 'FontWeight', 'bold');

% Subplot 1: Data Length vs Matching Correlation
nexttile; hold on;
for m = 1:numel(methods)
    method = methods{m};
    errorbar(durations, STATS.(conditions{1}).(method).matching_corr_mean, STATS.(conditions{1}).(method).matching_corr_std, ...
        '-o', 'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(m,:), 'DisplayName', method);
end
set(gca, 'XScale', 'log'); 
xticks(durations); xticklabels(string(durations));
xlabel('Data Length (seconds)'); ylabel('Mean Max Correlation (\rho)');
ylim([0 1]); title('Global Latent Matching Correlation');
grid on; legend('Location', 'best'); set(gca, 'FontSize', 18);

% Subplot 2: Data Length vs Spectral R^2
nexttile; hold on;
for m = 1:numel(methods)
    method = methods{m};
    errorbar(durations, STATS.(conditions{1}).(method).r2_mean, STATS.(conditions{1}).(method).r2_std, ...
        '-o', 'LineWidth', 2.5, 'MarkerSize', 8, 'Color', colors(m,:), 'DisplayName', method);
end
set(gca, 'XScale', 'log');
xticks(durations); xticklabels(string(durations));
xlabel('Data Length (seconds)'); ylabel('Mean Spectral R^2');
ylim([0 1]); title('Spectral R^2');
grid on; legend('Location', 'best'); set(gca, 'FontSize', 18);

saveas(fig1, fullfile(baseFolder, sprintf('DataLength_vs_Performance_k%d.png', k_range(1))));

% === FIGURE 2: Spectral R2 Frequency Profile per Duration ===
if nDurations > 1 && nDurations <= 6
    fig2 = figure('Position', [150, 150, 1600, 900]);
    nCols = ceil(sqrt(nDurations));
    nRows = ceil(nDurations / nCols);
    t2 = tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');
    sgtitle('Spectral R^2 vs. Peak Frequency across Data Lengths', 'FontSize', 24, 'FontWeight', 'bold');
    
    for dur_idx = 1:nDurations
        nexttile; hold on;
        for m = 1:numel(methods)
            method = methods{m};
            mu = STATS.(conditions{1}).(method).r2_per_latent_mean(dur_idx, :);
            sd = STATS.(conditions{1}).(method).r2_per_latent_std(dur_idx, :);
            
            x_col = reshape(double(param.f_peak), [], 1);
            y_col = reshape(double(mu), [], 1);
            e_col = reshape(double(sd), [], 1);
            errorbar(x_col, y_col, e_col, '-o', 'LineWidth', 2, ...
                     'Color', colors(m,:), 'DisplayName', method);
        end
        
        xlabel('Peak Frequency (Hz)'); ylabel('Spectral R^2');
        xticks(param.f_peak); ylim([0 1]);
        title(sprintf('Duration: %d sec', durations(dur_idx)));
        grid on; 
        if dur_idx == 1, legend('Location', 'best'); end
        set(gca, 'FontSize', 16);
    end
    saveas(fig2, fullfile(baseFolder, sprintf('FreqProfile_vs_DataLength_k%d.png', k_range(1))));
end

% === FIGURE 3: Data Length vs. Correlation per Latent Variable ===
fig3 = figure('Position', [200, 200, 1600, 900]);

nCols_lat = ceil(sqrt(param.N_F));
nRows_lat = ceil(param.N_F / nCols_lat);
t3 = tiledlayout(nRows_lat, nCols_lat, 'Padding', 'compact', 'TileSpacing', 'compact');
sgtitle('Data Length vs. Matching Correlation per True Latent', 'FontSize', 24, 'FontWeight', 'bold');

for f = 1:param.N_F
    nexttile; hold on;
    
    for m = 1:numel(methods)
        method = methods{m};
        
        % Extract mean and std for the specific true latent variable 'f'
        mu_f = STATS.(conditions{1}).(method).matching_corr_per_latent_mean(:, f);
        sd_f = STATS.(conditions{1}).(method).matching_corr_per_latent_std(:, f);
        
        x_col = reshape(double(durations), [], 1);
        y_col = reshape(double(mu_f), [], 1);
        e_col = reshape(double(sd_f), [], 1);
        
        errorbar(x_col, y_col, e_col, '-o', 'LineWidth', 2, ...
                 'MarkerSize', 6, 'Color', colors(m,:), 'DisplayName', method);
    end
    
    set(gca, 'XScale', 'log'); 
    xticks(durations); 
    xticklabels(string(durations));
    xlabel('Data Length (seconds)'); 
    ylabel('Max Correlation (\rho)');
    ylim([0 1]); 
    title(sprintf('Latent %d (%.1f Hz)', f, param.f_peak(f)));
    grid on; 
    
    if f == 1
        legend('Location', 'best');
    end
    set(gca, 'FontSize', 14);
end

saveas(fig3, fullfile(baseFolder, sprintf('DataLength_vs_MatchingCorr_PerLatent_k%d.png', k_range(1))));

% === FIGURE 4: Intrinsic Dimensionality vs. Data Length ===
fig4 = figure('Position', [300, 300, 900, 600]);
hold on;

% Plot 1: Ground Truth Reference Line
yline(param.N_F, '--k', sprintf('True Latents (N=%d)', param.N_F), ...
    'LineWidth', 2.5, 'LabelHorizontalAlignment', 'left', 'FontSize', 14, 'HandleVisibility', 'off');

% Plot 2: Full Rank
errorbar(durations, RESULTS.IntrinsicDim.rank_mean, RESULTS.IntrinsicDim.rank_std, ...
         '-o', 'LineWidth', 2.5, 'MarkerSize', 8, 'Color', [0.4 0.4 0.4], 'DisplayName', 'Matrix Rank');

% Plot 3: Marchenko-Pastur
errorbar(durations, RESULTS.IntrinsicDim.mp_mean, RESULTS.IntrinsicDim.mp_std, ...
         '-s', 'LineWidth', 2.5, 'MarkerSize', 8, 'Color', [0 0.4470 0.7410], 'DisplayName', 'Marchenko-Pastur');

% Plot 4: Velicer's MAP
errorbar(durations, RESULTS.IntrinsicDim.map_mean, RESULTS.IntrinsicDim.map_std, ...
         '-^', 'LineWidth', 2.5, 'MarkerSize', 8, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Velicer''s MAP');

set(gca, 'XScale', 'log');
xticks(durations); 
xticklabels(string(durations));
xlabel('Data Length (seconds)');
ylabel('Estimated Intrinsic Dimensionality');
title('Dimensionality Estimation vs. Data Length', 'FontSize', 22, 'FontWeight', 'bold');

% Set Y-limits to slightly above the max rank to give it breathing room
max_y = max([RESULTS.IntrinsicDim.rank_mean(:); param.N_F]);
ylim([0 max_y + 2]); 

grid on;
legend('Location', 'best');
set(gca, 'FontSize', 16);

saveas(fig4, fullfile(baseFolder, 'IntrinsicDimensionality_vs_DataLength.png'));

%% ---------------------------------------------------------------------
%  HELPER FUNCTIONS
% ---------------------------------------------------------------------
function parsave_struct(fname, s)
    % Helper to save a structure inside parfor
    save(fname, '-struct', 's');
end

function [num_factors, map_values] = velicer_map(data)
% VELICER_MAP Determines the number of factors to retain using Velicer's MAP test.
%
% Inputs:
%   data - An (n x p) matrix of raw data (n observations, p variables).
%
% Outputs:
%   num_factors - The optimal number of factors to retain.
%   map_values  - A vector containing the MAP values at each step (from 0 to p-1 factors).

    % 1. Compute the correlation matrix
    R = corrcoef(data);
    p = size(R, 1);
    
    % 2. Perform Eigenvalue Decomposition
    [eigvec, eigval_mat] = eig(R);
    eigval = diag(eigval_mat);
    
    % Sort eigenvalues and eigenvectors in descending order
    [eigval, idx] = sort(eigval, 'descend');
    eigvec = eigvec(:, idx);
    
    % Initialize array to store the MAP values
    map_values = zeros(1, p);
    
    % 3. Step 0: Calculate average squared correlation (no factors partialled out)
    R_off_diag = R - eye(p);
    map_values(1) = sum(R_off_diag(:).^2) / (p * (p - 1));
    
    % 4. Loop to partial out 1 to p-1 components
    for m = 1:(p-1)
        % Calculate the loadings for m components
        A = eigvec(:, 1:m) * diag(sqrt(eigval(1:m)));
        
        % Reproduce the correlation matrix from m components
        part_cov = R - (A * A');
        
        % Compute the partial correlation matrix
        d = diag(part_cov);
        inv_sqrt_d = diag(1 ./ sqrt(d));
        R_partial = inv_sqrt_d * part_cov * inv_sqrt_d;
        
        % Isolate off-diagonal elements
        R_partial_off = R_partial - eye(p);
        
        % Calculate the average squared partial correlation
        map_values(m+1) = sum(R_partial_off(:).^2) / (p * (p - 1));
    end
    
    % 5. Find the minimum MAP value
    % Note: MATLAB is 1-indexed, so index 1 represents 0 factors.
    [min_val, min_idx] = min(map_values);
    num_factors = min_idx - 1;
    
    % 6. Optional: Plot the results to visualize the minimum
    figure('Position',[50 50 800 500]);
    plot(0:(p-1), map_values, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
    hold on;
    plot(num_factors, map_values(min_idx), 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r'); % Highlight minimum
    coordinate_text = sprintf('  Min: (%d, %.4f)', num_factors, min_val);
    text(num_factors, 0.2, coordinate_text, ...
        'VerticalAlignment', 'top', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 26, ...
        'FontWeight', 'bold', ...
        'Color', 'r');
    xlabel('Number of Factors Retained');
    ylabel('Avg. Sqrd. Partial Corr.');
    title('Velicer''s MAP Test');
    legend('MAP Values', 'Minimum (Optimal Factors)');
    set(gca,'FontSize',24);
    grid on;
    hold off;
    
    % Output the result to the console
    fprintf('Velicer''s MAP Test recommends retaining %d factors.\n', num_factors);
end