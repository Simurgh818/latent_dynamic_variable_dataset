%% MAIN SCRIPT FOR DIMENSIONALITY REDUCTION BENCHMARK
clear; clc;

%% ----------------------------------------------------------
% 1. Load & Prepare Data
% ----------------------------------------------------------
eegFilename = 'simEEG_set4';
fullName = strcat(eegFilename, '.mat');
simEEG   = load(fullName);

s_eeg_like      = simEEG.train_sim_eeg_vals;
s_eeg_like_test = simEEG.test_sim_eeg_vals;
h_f   = simEEG.train_true_hF';

param.f_peak    = round([2 2.4 8 20 21 32 40 40],1);
fs_orig          = 1/simEEG.dt;
param.N_F       = size(simEEG.train_true_hF,1);

% Output path
baseFolder = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Dimensionality Reduction Review Paper'];

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
methods = {'PCA','AE', 'ICA','UMAP'}; 

results = struct();
for m = 1:numel(methods)
    results.(methods{m}).R2  = zeros(1, max_components);
    results.(methods{m}).MSE = zeros(1, max_components);
end

%% ----------------------------------------------------------
% 3. Loop through dimensionality reduction methods
% ----------------------------------------------------------
% java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment;
% warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
% 
% % Force headless Java mode (best chance in MATLAB)
% setenv('JAVA_TOOL_OPTIONS','-Djava.awt.headless=true');
% 
% % Also try MATLAB's internal headless toggle
% try
%     system_dependent('setheadless', true);
% catch
%     % Some versions don't support this; ignore gracefully
% end



% Start parallel pool (default: use all available cores)
% if isempty(gcp('nocreate'))
%     parpool;  
% end

for m = 1:numel(methods)
    method = methods{m};
    fprintf("Running %s...\n", method);
    R2_k_local = zeros(max_components,1);
    MSE_k_local = zeros(max_components,1);

    for k = component_range % par
        
        switch method
            
            case 'PCA'
                %% 1. Setup and Directories
                method_name = 'PCA';
                method_dir = fullfile(results_dir, method_name);
                [R2_test, MSE_test,outPCA] = runPCAAnalysis(eeg_train, eeg_test,...
                    H_train, H_test, param, k, fs_new, method_dir);
                R2_k_local(k) = R2_test(k);
                MSE_k_local(k) = MSE_test(k);

            case 'AE'
                % [R2_k, MSE_k] = runAutoencoderAnalysis(X_train, X_test, H_train, H_test, k);
                [R2_k_local(k), MSE_k_local(k), outAE] = runAutoencoderAnalysis(eeg_train, eeg_test,...
                    H_train, H_test, k, param, fs_new, results_dir);
            case 'ICA'
                % 1. Setup and Directories
                method_name = 'ICA';
                method_dir = fullfile(results_dir, method_name);
                [R2_k_local(k), MSE_k_local(k)] = runICAAnalysis(eeg_train, eeg_test, H_train, H_test, k, param, method_dir);

            case 'UMAP'
                % [R2_k, MSE_k] = runUMAPAnalysis(X_train, X_test, H_train, H_test, k);
                n_neighbors = 199;
                min_dist    = 0.3;
                [R2_k_local(k), MSE_k_local(k), outUMAP] = runUMAPAnalysis( ...
                    n_neighbors, min_dist, eeg_train, eeg_test, param, ...
                    H_train, H_test, k, param.fs, results_dir);


        end

    end
    results.(method).R2 = R2_k_local; 
    results.(method).MSE = MSE_k_local;
    R2_k_local = zeros(max_components,1);
    MSE_k_local = zeros(max_components,1);
end

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
legend(methods);
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
legend(methods);
grid on;
set(findall(gcf,'-property','FontSize'),'FontSize',16)
summary_trace_name = fullfile(results_dir, 'Main_Summary_Trace.png');
% summary_metrics_name = fullfile(results_dir, 'Main_Summary_Metrics.png');

saveas(fig1, summary_trace_name);
% saveas(fig3, summary_metrics_name);