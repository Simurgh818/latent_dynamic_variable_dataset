%% MAIN SCRIPT FOR DIMENSIONALITY REDUCTION BENCHMARK
clear; clc;

%% ----------------------------------------------------------
% 1. Load & Prepare Data
% ----------------------------------------------------------
% Paths
if exist('H:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Shared Code'];

    baseFolder = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Dimensionality Reduction Review Paper'];

elseif exist('G:\', 'dir')
    input_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Shared Code'];

    baseFolder = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Dimensionality Reduction Review Paper'];
else
    error('Unknown system: Cannot determine input and output paths.');
end

eegFilename = 'simEEG_set2_1_randF';
fullName = strcat(eegFilename, '.mat');
fullName_path = fullfile(input_dir,fullName);
simEEG   = load(fullName);

s_eeg_like      = simEEG.train_sim_eeg_vals;
s_eeg_like_test = simEEG.test_sim_eeg_vals;
h_f   = simEEG.train_true_hF';

param.f_peak    = round([40 40 32 21 20 8 2.4 2],1);
fs_orig          = 1/simEEG.dt;
param.N_F       = size(simEEG.train_true_hF,1);

% eegFilename = 'simEEG_set1';         % given EEG filename
subfolderName = ['results_' eegFilename];  % e.g., "results_simEEG_set1"

% Build full results directory path
results_dir = fullfile(baseFolder, subfolderName);
% Optionally sanitize filename (remove or replace illegal characters)
% illegal = '[<>:"/\\|?*]';
% results_dir = regexprep(results_dir, illegal, '_');

if ~exist(results_dir, 'dir')
    
    mkdir(results_dir);
end
%% Downsampling if needed
% original/high sampling rate
if fs_orig <=500
    fs_new  = fs_orig;               % desired analysis rate (Hz)
else
    fs_new = 500;
end
param.fs = fs_new;

% --- 1) Resample EEG-like signals (channels x time) ---
[nCh, T_orig_eeg] = size(s_eeg_like);
% Preallocate downsampled matrix: estimate new length
T_new_est = ceil(T_orig_eeg * fs_new / fs_orig);
s_eeg_ds = zeros(nCh, T_new_est);

for ch = 1:nCh
    % resample expects vector input, returns column vector
    y = resample(double(s_eeg_like(ch, :))', fs_new, fs_orig);  % Tnew x 1
    s_eeg_ds(ch, 1:length(y)) = y';
end
% trim to actual length
T_ds = size(y,1);
s_eeg_ds = s_eeg_ds(:, 1:T_ds);

% --- 2) Resample latent fields (time x N_F) ---
[T_hf, N_F] = size(h_f);
h_f_ds = zeros(ceil(T_hf * fs_new / fs_orig), N_F);
for fidx = 1:N_F
    yhf = resample(double(h_f(:, fidx)), fs_new, fs_orig); % col vector
    h_f_ds(1:length(yhf), fidx) = yhf;
end
h_f_ds = h_f_ds(1:length(yhf), :);  % enforce exact truncation (all columns same length)

% Normalize latent variables by their variance (per column)

h_f_normalized = h_f ./ std(h_f, 0, 1);  % Time × N_F
h_f_normalized_ds = h_f_ds ./ std(h_f_ds, 0, 1);  % Time × N_F



%% 
% Input features
eeg = s_eeg_like;

% Optional: normalize true h_f if needed
% h_f_normalized = normalize(h_f','zscore');

T = size(eeg,2);
idx_split = floor(0.8 * T);

eeg_train = eeg(:,1:idx_split);
eeg_test  = eeg(:,idx_split+1:end);

% Match dimensionality for true/target h_f
H_train = h_f_normalized(1:idx_split, :);
H_test  = h_f_normalized(idx_split+1:end, :);

%% ----------------------------------------------------------
% 2. Set number of components to test
% ----------------------------------------------------------
max_components = 10;       % or param-driven
component_range = 1:max_components;

% Store results: structure indexed by method name
methods = {'ICA','UMAP','AE'}; % 'PCA','dPCA',

results = struct();
for m = 1:numel(methods)
    results.(methods{m}).R2  = zeros(1, max_components);
    results.(methods{m}).MSE = zeros(1, max_components);
    results.(methods{m}).CORR = cell(1, max_components); 
end

%% ----------------------------------------------------------
% 3. Loop through dimensionality reduction methods
% ----------------------------------------------------------

% Start parallel pool (default: use all available cores)
% if isempty(gcp('nocreate'))
%     parpool;  
% end

outPCA.corr_PCA = table();
outDPCA.corr_dPCA = table();
outICA.corr_ICA = table();
outUMAP.corr_UMAP = table();
outAE.corr_AE = table();

f = param.f_peak(:);
[f_sorted, f_sortIdx] = sort(f, 'ascend');

for m = 1:numel(methods)
    method = methods{m};
    fprintf("Running %s...\n", method);
    R2_k_local = zeros(max_components,1);
    MSE_k_local = zeros(max_components,1);
    CORR_k_local = cell(max_components,1);

    method_dir = fullfile(results_dir, method);
    if ~exist(method_dir, 'dir')
        mkdir(method_dir);
    end
    for k = 7:7 % component_range % par
        
        switch method
            
            case 'PCA'
                %% 1. Setup and Directories
                method_name = 'PCA';
                method_dir = fullfile(results_dir, method_name);
                [R2_test, MSE_test,outPCA] = runPCAAnalysis(eeg_train, eeg_test,...
                    H_train, H_test, param, k, fs_new, method_dir);
                R2_k_local(k) = mean(R2_test(k,:));
                MSE_k_local(k) = mean(MSE_test(k,:));
                corr = outPCA.corr_PCA;
                R = outPCA.R_full;   
            case 'AE'
                % [R2_k, MSE_k] = runAutoencoderAnalysis(X_train, X_test, H_train, H_test, k);
                [R2_k_local(k), MSE_k_local(k), outAE] = runAutoencoderAnalysis(eeg_train, eeg_test,...
                    H_train, H_test, k, param, fs_new, results_dir);
                corr = outAE.corr_AE;
                R = outAE.R_full; 
            case 'ICA'
                % 1. Setup and Directories
                method_name = 'ICA';
                method_dir = fullfile(results_dir, method_name);
                [R2_k_local(k), MSE_k_local(k), outICA] = runICAAnalysis(eeg_train, eeg_test, H_train, H_test, k, param, method_dir);
                corr = outICA.corr_ICA;
                R = outICA.R_full; 
            case 'UMAP'
                % [R2_k, MSE_k] = runUMAPAnalysis(X_train, X_test, H_train, H_test, k);
                % javaFrame = feature('JavaFrame');
                java.lang.System.setProperty('java.awt.headless','false');   
                % % Reduce spurious Swing paint events
                % java.lang.System.setProperty('sun.awt.noerasebackground','true');

                n_neighbors = 100; % 199
                min_dist    = 0.3;
                [R2_k_local(k), MSE_k_local(k), outUMAP] = runUMAPAnalysis( ...
                    n_neighbors, min_dist, eeg_train, eeg_test, param, ...
                    H_train, H_test, k, param.fs, results_dir);
                corr = outUMAP.corr_UMAP;
                R = outUMAP.R_full; 
            case 'dPCA'
                method_name = 'dPCA';
                method_dir = fullfile(results_dir, method_name);
                [R2_test, MSE_test,outDPCA] = rundPCAAnalysis( ...
                    s_eeg_ds, h_f_normalized_ds, param, k, method_dir);
                R2_k_local(k) = mean(R2_test(k,:));
                MSE_k_local(k) = mean(MSE_test(k,:));
                corr = outDPCA.corr_dPCA;
                R = outDPCA.R_full; 
        end

        CORR_k_local{k} = corr;
        if isempty(corr), continue; end
        results.(method).summary(k).mean_corr = mean(corr{:,'corr_value'});

        fig = figure;   
        imagesc(R);
        clim([-1 1]); 
        axis square;
        xticks(1:size(R,2));
        yticks(1:size(R,1));
        xlabel('Component index (C)');
        ylabel('True latent index (h_f)');
        title(sprintf('%s: Latent–Component Correlation', method));
        set(gca, 'YDir','normal',...
            'TickLength',[0 0], ...
            'FontSize',14);
        colormap(parula);
        colorbar;
        % Save
        heatmap_name = fullfile(method_dir, ...
            sprintf('%s_CorrHeatmap_k%d.png', method, k));
        saveas(fig, heatmap_name);
        close(fig);

        % Ensure column vectors (defensive programming, EEG-style)
        c = corr.corr_value(:);
        
        % Sort by peak frequency (low → high)
        c_sorted = c(f_sortIdx);
        
        % Plot
        fig0 = figure;
        hold on;
        plot(f_sorted, c_sorted, '-o', 'LineWidth', 1.5,'DisplayName', method);
        xticks(unique(f_sorted));
        xticklabels(string(unique(f_sorted)));
        xlabel('Peak Frequencies (Hz)');
        ylabel('Correlation Value');
        title('Latent–Component Correlation Across Methods');
        grid on;
        ylim([0 1]);
        legend('Location','eastoutside');
        hold off;
        % Save and close
        cmp_name = fullfile(method_dir, 'CrossMethod_Corr_vs_Frequency.png');
        saveas(fig0, cmp_name);
        close(fig0);

    end
    results.(method).R2 = R2_k_local; 
    results.(method).MSE = MSE_k_local;
    results.(method).CORR = CORR_k_local;
  
    R2_k_local = zeros(max_components,1);
    MSE_k_local = zeros(max_components,1);
end


all_corr_tables = [outPCA.corr_PCA;
                   outDPCA.corr_dPCA;
                   outICA.corr_ICA;
                   outUMAP.corr_UMAP;
                   outAE.corr_AE];

fig_cmp = figure;
hold on;
for m = 1:numel(methods)
    method = methods{m};

    % Filter rows for this method
    tbl_m = all_corr_tables(strcmp(all_corr_tables.method, method), :);

    if isempty(tbl_m)
        continue
    end

    % Defensive alignment: ensure one row per latent
    % Assumes h_f column is 1..N_F
    c = nan(numel(f),1);
    c(tbl_m.h_f) = tbl_m.corr_value;

    % Sort correlations to match frequency order
    c_sorted = c(f_sortIdx);

    % Plot
    plot(f_sorted, c_sorted, '-o', ...
        'LineWidth', 1.8, ...
        'DisplayName', method);
end
xticks(unique(f_sorted));
xlabel('Peak Frequencies (Hz)');
ylabel('Correlation Value');
title('Latent–Component Correlation Across Methods');
ylim([0 1]);          % keep honest comparisons
grid on;
legend('Location','eastoutside');
set(gca, 'FontSize', 14);

% Save
cmp_name = fullfile(results_dir, 'CrossMethod_Corr_vs_Frequency.png');
saveas(fig_cmp, cmp_name);
% close(fig_cmp);

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
outFile = fullfile(results_dir, "Component_latentVariable_Corr.mat");
save(outFile, '-struct', 'results', 'summary')
%% ----------------------------------------------------------
% 4. Plot R^2 and MSE vs # components
% ----------------------------------------------------------
fig1 = figure;
tiledlayout(2,1);

% R2
nexttile;
hold on;
for m = 1:numel(methods)
    plot(component_range, results.(methods{m}).R2, 'LineWidth', 2);
end
xlabel('Number of Components');
ylim([0 1]);
ylabel('R^2');
title('R^2 vs Dimensionality');
legend(methods, 'Location','eastoutside');
grid on;

% MSE
nexttile;
hold on;
for m = 1:numel(methods)
    plot(component_range, results.(methods{m}).MSE, 'LineWidth', 2);
end
xlabel('Number of Components');
ylim([0 1]);
ylabel('MSE');
title('MSE vs Dimensionality');
legend(methods, 'Location','eastoutside');
grid on;
set(findall(gcf,'-property','FontSize'),'FontSize',16)
summary_trace_name = fullfile(results_dir, 'Main_Summary_Trace.png');
% summary_metrics_name = fullfile(results_dir, 'Main_Summary_Metrics.png');

saveas(fig1, summary_trace_name);
% saveas(fig3, summary_metrics_name);