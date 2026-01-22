%% MAIN SCRIPT FOR DIMENSIONALITY REDUCTION BENCHMARK
clear; clc; % close all;

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
nDatasets  = 1; % 5 10
k_range    = 6:7; % 18
nK         = numel(k_range);

% Store results: structure indexed by method name
methods = {'PCA'}; % 'dPCA','ICA', 'AE', 'UMAP'

EXP = struct();
param = struct();
% param.f_peak = round([1 4 8 12 30], 1);%2 5 10 13 20 25 30 50
param.duration = [1000];% 1, 5, 10, 60, 120, 600, 

% f = param.f_peak(:);
% [f_sorted, f_sortIdx] = sort(f, 'ascend');

% Best practice: Leave 1-2 cores free for the OS/Main Thread.
% For a 7-core system, 5 workers is a safe, high-performance choice.
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
    for d = 1:nDatasets
        fprintf('Dataset %d / %d (Worker Processing)\n', d, nDatasets);
        
        % --- 1. Load Data (Local to Worker) ---
        if d < 10 && ~strcmp(cond, 'ou') && ~strcmp(cond,'set4')
            eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, param.duration(1));
        elseif d<10
            eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, param.duration(1));
        elseif d == 1 && strcmp(cond, 'ou')
            eegFilename = sprintf('simEEG_Morrell_%s', cond);
        else
            eegFilename = sprintf('simEEG_%s_spat%d_dur%d', cond, d, param.duration(1));
        end
        
        % Load file
        loader = load(fullfile(input_dir, [eegFilename '.mat']));
        
        % Extract local variables
        s_eeg_like      = loader.train_sim_eeg_vals;
        % s_eeg_like_test = loader.test_sim_eeg_vals;
        h_f             = loader.train_true_hF';
        
        % Recalculate parameters locally
        local_param = loader.param; % Create a local copy of param
        
        fs_orig         = 1 / loader.dt;
        
        % Determine results directory for this dataset
        subfolderName = ['results_' eegFilename];
        local_results_dir = fullfile(baseFolder, subfolderName);
        if ~exist(local_results_dir, 'dir')
            mkdir(local_results_dir);
        end

        % --- 2. Pre-processing (Vectorized) ---
        if fs_orig <= 500
            fs_new = fs_orig;
        else
            fs_new = 500;
        end
        local_param.fs = fs_new;

        % Vectorized Resampling (Fast)
        % Transpose to (Time x Ch) for resample, then transpose back
        s_eeg_ds = resample(double(s_eeg_like)', fs_new, fs_orig)';
        
        % Resample latent fields
        h_f_ds_temp = resample(double(h_f), fs_new, fs_orig);
        h_f_ds = h_f_ds_temp(1:size(s_eeg_ds, 2),:); % Ensure length match
        
        % Normalize
        h_f_normalized_ds = h_f_ds ./ std(h_f_ds, 0, 1);
        
        % Split Train/Test
        eeg = s_eeg_like; 
        idx_split = floor(0.8 * size(eeg, 2));
        eeg_train = eeg(:, 1:idx_split);
        eeg_test  = eeg(:, idx_split+1:end);
        
        % Need normalized H for training? (Assuming yes based on previous code)
        h_f_norm_orig = h_f ./ std(h_f, 0, 1);
        H_train = h_f_norm_orig(1:idx_split, :);
        H_test  = h_f_norm_orig(idx_split+1:end, :);

        % --- 3. Method Loop ---
        % Initialize local result structure for this dataset
        dataset_res = struct();
        
        for m = 1:numel(methods)
            method = methods{m};
            method_dir = fullfile(local_results_dir, method);
            if ~exist(method_dir, 'dir'), mkdir(method_dir); end
            
            % Initialize storage arrays for this method
            dataset_res.(method).R2   = nan(1, nK);
            dataset_res.(method).MSE  = nan(1, nK);
            dataset_res.(method).CORR = cell(1, nK);
            dataset_res.(method).R_matrices = cell(1, nK);
            
            for ki = 1:nK
                k = k_range(ki);
                
                % Initialize generic temporary variables
                current_R2  = NaN;
                current_MSE = NaN;
                current_out = struct();
                current_corr_table = table(); % default empty
                current_R_matrix   = [];
                
                % Run Analysis
                switch method
                    case 'PCA'
                        [R2_test, MSE_test, outPCA] = runPCAAnalysis(eeg_train, eeg_test,...
                            H_train, H_test, local_param, k, method_dir);
                        
                        % Extract Scalar Values immediately
                        current_R2  = mean(R2_test(ki,:), 'omitnan');
                        current_MSE = mean(MSE_test(ki,:), 'omitnan');

                        current_out = outPCA;
                        if isfield(outPCA, 'corr_PCA'), current_corr_table = outPCA.corr_PCA; end
                        if isfield(outPCA, 'R_full'),   current_R_matrix   = outPCA.R_full;   end
                        
                    case 'AE'
                        [current_R2, current_MSE, outAE] = runAutoencoderAnalysis(eeg_train, eeg_test,...
                            H_train, H_test, k, local_param, local_results_dir);
                        current_out = outAE;
                        if isfield(outAE, 'corr_AE'), current_corr_table = outAE.corr_AE; end
                        if isfield(outAE, 'R_full'),  current_R_matrix   = outAE.R_full;  end

                    case 'ICA'
                        [current_R2, current_MSE, outICA] = runICAAnalysis(eeg_train, eeg_test, ...
                             H_train, H_test, k, local_param, method_dir);
                        current_out = outICA;
                        if isfield(outICA, 'corr_ICA'), current_corr_table = outICA.corr_ICA; end
                        if isfield(outICA, 'R_full'),   current_R_matrix   = outICA.R_full;   end

                    case 'UMAP'
                        % Note: Java properties should ideally be set outside parfor, 
                        % but some workers might need it reset.
                        n_neighbors = 3; min_dist = 0.99;
                        [current_R2, current_MSE, outUMAP] = runUMAPAnalysis( ...
                            n_neighbors, min_dist, eeg_train, eeg_test, local_param, ...
                            H_train, H_test, k, local_results_dir);
                        current_out = outUMAP;
                        if isfield(outUMAP, 'corr_UMAP'), current_corr_table = outUMAP.corr_UMAP; end
                        if isfield(outUMAP, 'R_full'),    current_R_matrix   = outUMAP.R_full;    end

                    case 'dPCA'
                        [R2_test, MSE_test, outDPCA] = rundPCAAnalysis( ...
                            s_eeg_ds, h_f_normalized_ds, local_param, k, method_dir);
                        current_R2  = mean(R2_test(ki,:), 'omitnan');
                        current_MSE = mean(MSE_test(ki,:), 'omitnan');
        
                        current_out = outDPCA;
                        if isfield(outDPCA, 'corr_dPCA'), current_corr_table = outDPCA.corr_dPCA; end
                        if isfield(outDPCA, 'R_full'),    current_R_matrix   = outDPCA.R_full;    end
                end
                
                % --- Store Data (No Plotting) ---
                dataset_res.(method).R2(ki)  = current_R2;
                dataset_res.(method).MSE(ki) = current_MSE;
                
                % Process Correlation Table Metadata
                if ~isempty(current_corr_table)
                    current_corr_table.method  = repmat(string(method), height(current_corr_table), 1);
                    current_corr_table.dataset = repmat(d, height(current_corr_table), 1);
                    current_corr_table.k       = repmat(k, height(current_corr_table), 1);
                end
                
                dataset_res.(method).CORR{ki}       = current_corr_table;
                dataset_res.(method).R_matrices{ki} = current_R_matrix;
            end
        end
        
        % Save results for this dataset into the cell array
        dataset_results{d} = dataset_res;
        
    end % End Parfor

    % ---------------------------------------------------------------------
    % POST-PROCESSING & PLOTTING (Serial)
    % ---------------------------------------------------------------------
    % Unpack results back into EXP structure and Generate Plots
    for d = 1:nDatasets
        EXP.(cond).dataset(d) = dataset_results{d};
        
        % Re-define paths for saving plots
        if d < 10 && ~strcmp(cond, 'ou') && ~strcmp(cond,'set4')
            eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, local_param.T(1));
        elseif d<10
            eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, local_param.T(1));
        elseif d == 1 && strcmp(cond, 'ou')
            eegFilename = sprintf('simEEG_Morrell_%s', cond);
        else
            eegFilename = sprintf('simEEG_%s_spat%d_dur%d', cond, d, local_param.T(1));
        end
        subfolderName = ['results_' eegFilename];
        local_results_dir = fullfile(baseFolder, subfolderName);
        
        for m = 1:numel(methods)
            method = methods{m};
            method_dir = fullfile(local_results_dir, method);
            
            % Generate Heatmaps (now safe in serial)
            for ki = 1:nK
                k = k_range(ki);
                R = EXP.(cond).dataset(d).(method).R_matrices{ki};
                
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

        for d = 1:nDatasets
            R2_all(d,:)  = EXP.(cond).dataset(d).(method).R2;
            MSE_all(d,:) = EXP.(cond).dataset(d).(method).MSE;
        end

        STATS.(cond).(method).R2.mean  = mean(R2_all,1,'omitnan');
        STATS.(cond).(method).R2.std   = std(R2_all,0,1,'omitnan');

        STATS.(cond).(method).MSE.mean = mean(MSE_all,1,'omitnan');
        STATS.(cond).(method).MSE.std  = std(MSE_all,0,1,'omitnan');
    end
end


% all_corr_tables = [outPCA.corr_PCA;
%                    outDPCA.corr_dPCA;
%                    outICA.corr_ICA;
%                    outUMAP.corr_UMAP;
%                    outAE.corr_AE];
CORR_STATS = struct();

for c = 1:numel(conditions)
    cond = conditions{c};

    for m = 1:numel(methods)
        method = methods{m};

        for ki = 1:nK
            % Collect corr tables from all datasets
            corr_mat = nan(nDatasets, size(local_param.f_peak,2));

            for d = 1:nDatasets
                tbl = EXP.(cond).dataset(d).(method).CORR{ki};

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


% fig_cmp = figure;
% hold on;
% for m = 1:numel(methods)
%     method = methods{m};
% 
%     % Filter rows for this method
%     tbl_m = all_corr_tables(strcmp(all_corr_tables.method, method), :);
% 
%     if isempty(tbl_m)
%         continue
%     end
% 
%     % Defensive alignment: ensure one row per latent
%     % Assumes h_f column is 1..N_F
%     c = nan(numel(f),1);
%     c(tbl_m.h_f) = tbl_m.corr_value;
% 
%     % Sort correlations to match frequency order
%     c_sorted = c(f_sortIdx);
% 
%     % Plot
%     plot(f_sorted, c_sorted, '-o', ...
%         'LineWidth', 1.8, ...
%         'DisplayName', method);
% end
% xticks(unique(f_sorted));
% xlabel('Peak Frequencies (Hz)');
% ylabel('Correlation Value');
% title('Latent–Component Correlation Across Methods');
% ylim([0 1]);          % keep honest comparisons
% grid on;
% legend('Location','eastoutside');
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

            % mu_sorted = mu(f_sortIdx);
            % sd_sorted = sd(f_sortIdx);

            errorbar(local_param.f_peak, mu, sd, '-o', ...
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



% Save
% cmp_name = fullfile(local_results_dir, 'CrossMethod_Corr_vs_Frequency.png');
% saveas(fig_cmp, cmp_name);
% close(fig_cmp);

CORR_ALL = struct();

for c = 1:numel(conditions)
    cond = conditions{c};

    CORR_ALL.(cond).dataset = struct();  % force struct array

    for d = 1:nDatasets
        CORR_ALL.(cond).dataset(d).tbl = table();
    end
end

CANON_VARS = {
    'corr_value', ...
    'h_f', ...
    'component', ...
    'method', ...
    'dataset', ...
    'k'
};


for c = 1:numel(conditions)
    cond = conditions{c};

    for d = 1:nDatasets
        tbl_d = table();

        for m = 1:numel(methods)
            method = methods{m};

            for ki = 1:nK
                corr = EXP.(cond).dataset(d).(method).CORR{ki};

                if isempty(corr)
                    continue
                end

                % --- Enforce canonical schema -----------------------------------------
                if isempty(tbl_d)
                    % First non-empty table defines the template
                    tbl_d = corr(:, intersect(CANON_VARS, corr.Properties.VariableNames));
                else
                    % Add missing variables to corr
                    missing = setdiff(tbl_d.Properties.VariableNames, corr.Properties.VariableNames);
                    for v = missing
                        corr.(v{1}) = nan(height(corr),1);
                    end
                
                    % Add missing variables to tbl_d (rare but safe)
                    missing = setdiff(corr.Properties.VariableNames, tbl_d.Properties.VariableNames);
                    for v = missing
                        tbl_d.(v{1}) = nan(height(tbl_d),1);
                    end
                
                    % Reorder corr to match tbl_d
                    corr = corr(:, tbl_d.Properties.VariableNames);
                
                    % Concatenate
                    tbl_d = [tbl_d; corr];
                end

            end
        end

        CORR_ALL.(cond).dataset(d).tbl = tbl_d;

    end
end


all_corr_tables = table();

for c = 1:numel(conditions)
    cond = conditions{c};

    for d = 1:nDatasets
        tbl = CORR_ALL.(cond).dataset(d).tbl;

        if isempty(tbl), continue; end

        tbl.condition = repmat(string(cond), height(tbl), 1);
        all_corr_tables = [all_corr_tables; tbl];
    end
end


summary = groupsummary(all_corr_tables, 'method', 'mean', 'corr_value');
summary_min = groupsummary(all_corr_tables, 'method', 'min', 'corr_value');
threshold = 0.46;
good_counts = groupsummary( ...
                all_corr_tables(all_corr_tables.corr_value > threshold,:), ...
                'method',@sum,'corr_value');
results.summary.mean = summary;
results.summary.min  = summary_min;
results.summary.good_counts = good_counts;
results.summary.threshold = threshold;

% Save file -------------------------------------------------------------
outFile = fullfile(local_results_dir, "Component_latentVariable_Corr.mat");
save(outFile, '-struct', 'results', 'summary')
%% ----------------------------------------------------------
% 4. Plot R^2 and MSE vs # components
% ----------------------------------------------------------
fig1 = figure;
tiledlayout(2,1);

% R2
nexttile;
% hold on;
% for m = 1:numel(methods)
%     plot(component_range, results.(methods{m}).R2, 'LineWidth', 2);
% end
% xlabel('Number of Components');
% ylim([0 1]);
% ylabel('R^2');
% title('R^2 vs Dimensionality');
% legend(methods, 'Location','eastoutside');
% grid on;
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
    ylabel('R^2');
    ylim([0 1]);
    title(sprintf('R^2 vs k (%s)', cond));
    grid on;
    legend('Location','eastoutside');
    set(gca, 'FontSize', 16);
end


% MSE
nexttile;
% hold on;
% for m = 1:numel(methods)
%     plot(component_range, results.(methods{m}).MSE, 'LineWidth', 2);
% end
% xlabel('Number of Components');
% ylim([0 1]);
% ylabel('MSE');
% title('MSE vs Dimensionality');
% legend(methods, 'Location','eastoutside');
% grid on;
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
    ylabel('MSE^2');
    ylim([0 1]);
    title(sprintf('MSE^2 vs k (%s)', cond));
    grid on;
    legend('Location','eastoutside');
    set(gca, 'FontSize', 16);
end

set(findall(gcf,'-property','FontSize'),'FontSize',16)
summary_trace_name = fullfile(local_results_dir, 'Main_Summary_Trace.png');
% summary_metrics_name = fullfile(results_dir, 'Main_Summary_Metrics.png');

saveas(fig1, summary_trace_name);
% saveas(fig3, summary_metrics_name);