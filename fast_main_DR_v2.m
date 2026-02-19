%% MAIN SCRIPT FOR DIMENSIONALITY REDUCTION BENCHMARK
clear; clc; close all;

%% ----------------------------------------------------------
% 1. Load & Prepare Data
% ----------------------------------------------------------
% Paths
if exist('H:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Shared Code' filesep 'simEEG']; %  filesep 'diffDuration'

    baseFolder = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Dimensionality Reduction Review Paper'];

elseif exist('G:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Shared Code' filesep 'simEEG']; %  filesep 'diffDuration'

    baseFolder = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Dimensionality Reduction Review Paper'];
else
    error('Unknown system: Cannot determine input and output paths.');
end

%% Loop through experiments

conditions = {'set4'}; %,'ou', 'set2',  linear, nonlinear
nDatasets  = 1; % 10
k_range    = 6:6; %5 8
nK         = numel(k_range);

% Store results: structure indexed by method name
methods = {'UMAP'}; %'PCA', 'AE','dPCA', 'ICA','UMAP' 

EXP = struct();
param = struct();
% param.f_peak = round([1 4 8 12 30], 1);%2 5 10 13 20 25 30 50
param.duration = [1000];% 1, 5, 10, 60, 120, 600, 

RESULTS = struct();

RESULTS.meta = struct();
RESULTS.meta.created = datetime;
RESULTS.meta.description = "Dimensionality reduction benchmark";

% Best practice: Leave 1-2 cores free for the OS/Main Thread.
% For a 7-core system, 5 workers is a safe, high-performance choice.
% target_workers = 5; 
% current_pool = gcp('nocreate');
% 
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
    for d = 1:nDatasets
        fprintf('Dataset %d / %d (Worker Processing)\n', d, nDatasets);
        data = struct();
        
        % --- 1. Load Data (Local to Worker) ---
        if d < 10 && ~strcmp(cond, 'ou') && ~strcmp(cond,'set4')
            eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, param.duration(1));
        elseif d < 10
            eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, param.duration(1));
        elseif d == 1 && strcmp(cond, 'ou')
            eegFilename = sprintf('simEEG_Morrell_%s', cond);
        else
            eegFilename = sprintf('simEEG_%s_spat%d_dur%d', cond, d, param.duration(1));
        end
        dataset_name = eegFilename;
        
        % Load file
        loader = load(fullfile(input_dir, [eegFilename '.mat']));
        
        % Extract local variables
        s_eeg_like      = loader.train_sim_eeg_vals;
        h_f             = loader.train_true_hF';
        
        % Recalculate parameters locally
        local_param = loader.param; 
        fs_orig         = 1 / loader.dt;
        data.fs_orig = fs_orig;
        
        % Determine results directory for this dataset
        subfolderName = ['results_' eegFilename];
        local_results_dir = fullfile(baseFolder, subfolderName);
        if ~exist(local_results_dir, 'dir')
            mkdir(local_results_dir);
        end
        
        % --- 2. Pre-processing (Vectorized) ---
        if data.fs_orig <= 500
            data.fs_new = data.fs_orig;
        else
            data.fs_new = 500;
        end
        local_param.fs = data.fs_new;
        
        % Vectorized Resampling
        s_eeg_ds = resample(double(s_eeg_like)', data.fs_new, data.fs_orig)';
        
        h_f_ds_temp = resample(double(h_f), data.fs_new, data.fs_orig);
        h_f_ds = h_f_ds_temp(1:size(s_eeg_ds, 2),:); 
        
        h_f_normalized_ds = h_f_ds ./ std(h_f_ds, 0, 1);
        
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
        data.eeg_ds    = s_eeg_ds; 
        data.H_ds      = h_f_normalized_ds;
        
        % --- 3. Method Loop ---
        dataset_res = struct();
        dataset_res.f_peak = local_param.f_peak;
        
        % Initialize a table to collect ALL entries for this dataset
        all_d_entries = table();
        
        for m = 1:numel(methods)
            method = methods{m};
            method_dir = fullfile(local_results_dir, method);
            if ~exist(method_dir, 'dir'), mkdir(method_dir); end
            
            dataset_res.(method).R2   = nan(1, nK);
            dataset_res.(method).MSE  = nan(1, nK);
            dataset_res.(method).CORR = cell(1, nK);
            dataset_res.(method).R_matrices = cell(1, nK);
            dataset_res.(method).spectral_R2 = nan(local_param.N_F, nK);

            for ki = 1:nK
                k = k_range(ki);
              
                entry = runDimRedMethod( ...
                    method, data, local_param, k, ki, cond, dataset_name, method_dir,...
                    local_results_dir);
                
                % Store basic stats
                dataset_res.(method).R2(ki)  = entry.stats.R2;
                dataset_res.(method).MSE(ki) = entry.stats.MSE;                             
                dataset_res.(method).CORR{ki} = entry.corr;
                dataset_res.(method).R_matrices{ki} = entry.R_matrix;
                dataset_res.(method).spectral_R2(:,ki) = entry.spectral_R2;

                if ki== nK
                    dataset_res.(method).h_recon_test = entry.out.h_recon_test;
                end
                
                % --- Fix for Duplicate Variable Names ---
                current_corr_table = entry.corr; 
                
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
                    
                    % RESULTS.entries = [RESULTS.entries; entry];

                    % Append to the main collector
                    all_d_entries = [all_d_entries; current_corr_table];
                end
            end
        end

        % saving out snippets of data 
        % To Do: take h_test sample
        snippet_seconds = 10;
        snippet_samples = min(size(data.H_ds, 1), round(snippet_seconds * local_param.fs));
        time_idx = 1:snippet_samples;

        snippets = struct();
        snippets.time_vector = (time_idx - 1) / local_param.fs;
        snippets.H_test = data.H_test(time_idx, :); % Ground truth, if we need to save more
        % space we can cast as single() since double is 64-bit and single
        % 32-bit.
        snippets.eeg_train = data.eeg_train;
        snippets.param = local_param;
        
        % Collect the reconstruction from each method at the max k
        for m = 1:numel(methods)
            method = methods{m};
            % h_recon_test is already in dataset_res from the runMethod function
            % Ensure we only take the snippet window
            snippets.(method).h_recon_test = dataset_res.(method).h_recon_test(time_idx, :);% if we need to save more
            % space we can cast as single() since double is 64-bit and single
            % 32-bit.
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
                R = EXP.(cond).dataset(d).analysis.(method).R_matrices{ki};
                
                if isempty(R), continue; end
                
                fig = figure; % ('Visible', 'off')  
                imagesc(R);
                clim([-1 1]); 
                axis square;
                xlabel('Component index (C)'); ylabel('True latent index (h_f)');
                title(sprintf('%s: Latent–Component Correlation', method));
                colormap(parula); colorbar;
                saveas(fig, fullfile(method_dir, sprintf('%s_CorrHeatmap_k%d.png', method, k)));
                close(fig);
            end
            
            % Generate Frequency Plots
            % (Insert your frequency plotting code here using EXP data)
        end
        
    end
end


STATS = struct();

for c = 1:numel(conditions)
    cond = conditions{c};

    for m = 1:numel(methods)
        method = methods{m};

        R2_all  = nan(nDatasets, nK);
        MSE_all = nan(nDatasets, nK);
        spectral_R2_all = nan(nDatasets, local_param.N_F, nK);

        for d = 1:nDatasets
            R2_all(d,:)  = EXP.(cond).dataset(d).analysis.(method).R2;
            MSE_all(d,:) = EXP.(cond).dataset(d).analysis.(method).MSE;
            spectral_R2_all(d, :, :) = EXP.(cond).dataset(d).analysis.(method).spectral_R2;
        end

        STATS.(cond).(method).R2.mean  = mean(R2_all,1,'omitnan');
        STATS.(cond).(method).R2.std   = std(R2_all,0,1,'omitnan');

        STATS.(cond).(method).MSE.mean = mean(MSE_all,1,'omitnan');
        STATS.(cond).(method).MSE.std  = std(MSE_all,0,1,'omitnan');

        STATS.(cond).(method).spectral_R2.mean  = squeeze(mean(spectral_R2_all, 1, 'omitnan'));
        STATS.(cond).(method).spectral_R2.std   = squeeze(std(spectral_R2_all, 0, 1, 'omitnan'));
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
                tbl = EXP.(cond).dataset(d).analysis.(method).CORR{ki};

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
        ylabel('Correlation');
        ylim([0 1]);
        title(sprintf('Latent–Component Corr (k=%d, %s)', k, cond));
        grid on;
        legend('Location','eastoutside');
        set(gca, 'FontSize', 16);
        corr_figure_name = fullfile(local_results_dir, sprintf('Latent–Component Corr %s k_%d.png', cond, k));
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

summary = groupsummary(all_corr_tables, 'method', 'mean', 'corr_value');
summary_min = groupsummary(all_corr_tables, 'method', 'min', 'corr_value');
threshold = 0.39; % since UMAP best performance is 0.4
good_counts = groupsummary( ...
                all_corr_tables(all_corr_tables.corr_value > threshold,:), ...
                'method',@sum,'corr_value');
results.summary.mean = summary;
results.summary.min  = summary_min;
results.summary.good_counts = good_counts;
results.summary.threshold = threshold;

% Save file -------------------------------------------------------------
save(fullfile(local_results_dir, "RESULTS.mat"), "RESULTS", "-v7.3");
save(fullfile(local_results_dir, "Component_latentVariable_Corr.mat"), ...
     "all_corr_tables", "results");

%% ----------------------------------------------------------
% 4. Plot R^2 and MSE vs # components
% ----------------------------------------------------------
fig1 = figure;
tiledlayout(2,1);

% R2
nexttile;
for c = 1:numel(conditions)
    cond = conditions{c};

    hold on;
    colors = lines(numel(methods));

    for m = 1:numel(methods)
        method = methods{m};
        mu = STATS.(cond).(method).R2.mean;
        sd = STATS.(cond).(method).R2.std;

        errorbar(k_range, mu, sd, '-o', ...
            'LineWidth',2, ...
            'Color',colors(m,:), ...
            'DisplayName',method);
    end

    xlabel('Number of Components');
    xticks(linspace(min(k_range), max(k_range), nK));
    ylabel('R^2');
    ylim([0 1]);
    title(sprintf('R^2 vs k (%s)', cond));
    grid on;
    legend('Location','eastoutside');
    set(gca, 'FontSize', 16);
end


% MSE
nexttile;
for c = 1:numel(conditions)
    cond = conditions{c};

    hold on;
    colors = lines(numel(methods));

    for m = 1:numel(methods)
        method = methods{m};
        mu = STATS.(cond).(method).MSE.mean;
        sd = STATS.(cond).(method).MSE.std;

        errorbar(k_range, mu, sd, '-o', ...
            'LineWidth',2, ...
            'Color',colors(m,:), ...
            'DisplayName',method);
    end

    xlabel('Number of Components');
    xticks(linspace(min(k_range), max(k_range), nK));
    ylabel('MSE^2');
    ylim([0 1]);
    title(sprintf('MSE^2 vs k (%s)', cond));
    grid on;
    legend('Location','eastoutside');
    set(gca, 'FontSize', 16);
end

set(findall(gcf,'-property','FontSize'),'FontSize',16)
summary_trace_name = fullfile(local_results_dir, 'Main_Summary_Trace.png');
saveas(fig1, summary_trace_name);
%% ---------------------------------------------------------------------
%  HELPER FUNCTIONS
% ---------------------------------------------------------------------
function parsave_struct(fname, s)
    % Helper to save a structure inside parfor
    save(fname, '-struct', 's');
end