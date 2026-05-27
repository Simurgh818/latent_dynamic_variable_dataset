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
methods    = {'PCA','AE','ICA'}; % Only run PCA and AE
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

%% ---------------------------------------------------------------------
%  HELPER FUNCTIONS
% ---------------------------------------------------------------------
function parsave_struct(fname, s)
    % Helper to save a structure inside parfor
    save(fname, '-struct', 's');
end