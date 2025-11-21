%% MAIN SCRIPT FOR DIMENSIONALITY REDUCTION BENCHMARK
clear; clc;

%% ----------------------------------------------------------
% 1. Load & Prepare Data
% ----------------------------------------------------------
simEEG = load("simEEG_set1.mat");

s_eeg_like      = simEEG.train_sim_eeg_vals;
s_eeg_like_test = simEEG.test_sim_eeg_vals;
h_f_processed   = simEEG.train_true_hF';

param.f_peak    = [1,2,3,4,5,6,7,8];
fs_new          = 1/simEEG.dt;
fs_orig         = fs_new;
param.N_F       = size(simEEG.train_true_hF,1);

% Input features
eeg = s_eeg_like;

% Optional: normalize true h_f if needed
% h_f_normalized = normalize(h_f_processed','zscore');

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
methods = {'PCA','ICA','UMAP','AE'}; 

results = struct();
for m = 1:numel(methods)
    results.(methods{m}).R2  = zeros(1, max_components);
    results.(methods{m}).MSE = zeros(1, max_components);
end

%% ----------------------------------------------------------
% 3. Loop through dimensionality reduction methods
% ----------------------------------------------------------
for m = 1:numel(methods)
    method = methods{m};
    fprintf("Running %s...\n", method);

    for k = component_range
        switch method
            
            case 'PCA'
                % Example function signature:
                % [R2_k, MSE_k] = runPCAAnalysis(X_train, X_test, H_train, H_test, k);
                [R2_test, MSE_test,outPCA] = runPCAAnalysis(eeg_train, eeg_test,...
                    H_train, H_test, k, results_dir);

            % case 'ICA'
            %     % [R2_k, MSE_k] = runICAAnalysis(X_train, X_test, H_train, H_test, k);
            %     [R2_k, MSE_k] = runICAAnalysis(X_train, X_test, H_train, H_test, k);
            % 
            % case 'UMAP'
            %     % [R2_k, MSE_k] = runUMAPAnalysis(X_train, X_test, H_train, H_test, k);
            %     [R2_k, MSE_k] = runUMAPAnalysis(X_train, X_test, H_train, H_test, k);
            % 
            % case 'AE'
            %     % [R2_k, MSE_k] = runAutoencoderAnalysis(X_train, X_test, H_train, H_test, k);
            %     [R2_k, MSE_k] = runAutoencoderAnalysis(X_train, X_test, H_train, H_test, k);

        end

        results.(method).R2(k)  = R2_k;
        results.(method).MSE(k) = MSE_k;
    end
end

%% ----------------------------------------------------------
% 4. Plot R^2 and MSE vs # components
% ----------------------------------------------------------
figure;
tiledlayout(2,1);

% R2
nexttile;
hold on;
for m = 1:numel(methods)
    plot(component_range, results.(methods{m}).R2, 'LineWidth', 2);
end
xlabel('Number of Components');
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
ylabel('MSE');
title('MSE vs Dimensionality');
legend(methods);
grid on;

