%% Spatial Overlap Hypothesis Testing: Pairwise Analysis
clear; clc; close all;

% =========================================================================
% 1. DEFINE PATHS AND PARAMETERS
% =========================================================================
% Using your G: drive paths (adjust to H: or I: if you are on a different machine)
if exist('G:\', 'dir')
    data_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'simEEG'];
    
    results_dir = ['C:' filesep 'Users' filesep 'sdabiri' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'Results'];
else
    % Fallback to I: drive paths
    data_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'simEEG'];
    
    results_dir = ['C:' filesep 'Users' filesep 'sinad' filesep ...
    'OneDrive - Georgia Institute of Technology' filesep ...
    'Dr. Sederberg MaTRIX Lab' filesep ...
    'Method Paper' filesep 'Results'];
end

% Experiment Parameters
cond = 'set4';
target_duration = 1000;
nDatasets = 10; 

% Because you tested multiple k's, choose which k to evaluate spatial overlap on!
k_range = 5:9;
target_k = 6; 
ki_target = find(k_range == target_k);

% Pre-allocate aggregate arrays
all_pair_overlap = [];
all_diff_PCA = [];
all_diff_AE = [];

% =========================================================================
% 2. LOOP THROUGH DATASETS: EXTRACT TRUE MAPS & LOCAL RESULTS
% =========================================================================
fprintf('Extracting spatial mixing matrices and latent statistics for k=%d...\n', target_k);

for d = 1:nDatasets
    % --- Determine the base filename for this dataset ---
    if d < 10 
        eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, target_duration);
    else
        eegFilename = sprintf('simEEG_%s_spat%d_dur%d', cond, d, target_duration);
    end
    
    % --- A. Load the Ground Truth Spatial Mixing Matrix (W_true) ---
    key_filename = sprintf('%s_key.mat', eegFilename);
    key_filepath = fullfile(data_dir, key_filename);
    
    if ~exist(key_filepath, 'file')
        warning('Cannot find key file %s. Skipping dataset %d.', key_filename, d);
        continue;
    end
    
    % Load all variables in the key file to find the mixing matrix
    key_data = load(key_filepath);
    
    % Based on your synthetic generator, the matrix is likely called spatial_comps
    if isfield(key_data, 'spatial_comps')
        W_true = key_data.spatial_comps;
    elseif isfield(key_data, 'W')
        W_true = key_data.W;
    else
        error('Dataset %d: Could not find "spatial_comps" in the key file. Please check what the mixing matrix is named!', d);
    end
    
    % --- B. Load this Dataset's Individual Results File ---
    subfolderName = ['results_' eegFilename];
    ds_filename = fullfile(results_dir, subfolderName, sprintf('Results_%s.mat', eegFilename));
    
    if ~exist(ds_filename, 'file')
        warning('Cannot find results file %s. Skipping dataset %d.', ds_filename, d);
        continue;
    end
    
    res = load(ds_filename);
    
    % --- C. Extract Recovery Correlations for the Target k ---
    % PCA
    pca_tbl = res.analysis.PCA.Comp_latent_matching_corr{ki_target};
    pca_tbl = sortrows(pca_tbl, 'h_f'); 
    pca_corr = pca_tbl.corr_value;
    
    % Autoencoder
    ae_tbl = res.analysis.AE.Comp_latent_matching_corr{ki_target};
    ae_tbl = sortrows(ae_tbl, 'h_f');
    ae_corr = ae_tbl.corr_value;
    
    % --- D. Calculate Pairwise Metrics for this Dataset ---
    pairs_PCA = calculatePairwiseSimilarities(W_true, pca_corr);
    pairs_AE  = calculatePairwiseSimilarities(W_true, ae_corr);
    
    % --- E. Aggregate across all datasets ---
    all_pair_overlap = [all_pair_overlap; pairs_PCA.overlap]; 
    all_diff_PCA     = [all_diff_PCA; pairs_PCA.perf_diff];
    all_diff_AE      = [all_diff_AE; pairs_AE.perf_diff];
end

% =========================================================================
% 3. STATISTICAL ANALYSIS & PLOTTING
% =========================================================================
fprintf('Fitting models and plotting results...\n');

% Fit Linear Regression Models
mdl_PCA = fitlm(all_pair_overlap, all_diff_PCA);
mdl_AE  = fitlm(all_pair_overlap, all_diff_AE);

% Visualization
figure('Position', [100, 100, 1200, 500], 'Name', sprintf('Pairwise Latent Similarity (k=%d)', target_k));

% Subplot 1: PCA
subplot(1,2,1);
scatter(all_pair_overlap, all_diff_PCA, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(mdl_PCA.Variables.x1, mdl_PCA.Fitted, 'k-', 'LineWidth', 2);
xlabel('Pairwise Spatial Overlap (Cosine Similarity)');
ylabel('Absolute Difference in Recovery (\Delta Corr)');
title('PCA: Performance Convergence');
ylim([-0.05 1]); xlim([0 1.05]);
grid on;
% Add R^2 and p-value
txt_pca = sprintf('R^2 = %.3f\np = %.3e', mdl_PCA.Rsquared.Ordinary, mdl_PCA.Coefficients.pValue(2));
text(0.65, 0.85, txt_pca, 'FontSize', 12, 'BackgroundColor', 'w', 'EdgeColor', 'k');

% Subplot 2: Autoencoder
subplot(1,2,2);
scatter(all_pair_overlap, all_diff_AE, 50, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot(mdl_AE.Variables.x1, mdl_AE.Fitted, 'k-', 'LineWidth', 2);
xlabel('Pairwise Spatial Overlap (Cosine Similarity)');
ylabel('Absolute Difference in Recovery (\Delta Corr)');
title('Autoencoder: Performance Convergence');
ylim([-0.05 1]); xlim([0 1.05]);
grid on;
% Add R^2 and p-value
txt_ae = sprintf('R^2 = %.3f\np = %.3e', mdl_AE.Rsquared.Ordinary, mdl_AE.Coefficients.pValue(2));
text(0.65, 0.85, txt_ae, 'FontSize', 12, 'BackgroundColor', 'w', 'EdgeColor', 'k');
