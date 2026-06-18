%% Spatial Overlap Hypothesis Testing: Pairwise Analysis (Difference & Mean)
clear; clc; close all;

% =========================================================================
% 1. DEFINE PATHS AND PARAMETERS
% =========================================================================
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

% Target component limit
k_range = 5:9;
target_k = 6; 
ki_target = find(k_range == target_k);

% Pre-allocate aggregate arrays
all_pair_overlap = [];
all_diff_PCA = [];
all_diff_AE = [];
all_mean_PCA = []; 
all_mean_AE = [];

% --- NEW VARIABLES FOR TRACKING IDENTITIES ---
all_dataset_id = [];
all_freq_A = [];
all_freq_B = [];

% =========================================================================
% 2. LOOP THROUGH DATASETS
% =========================================================================
fprintf('Extracting spatial mixing matrices and latent statistics for k=%d...\n', target_k);

for d = 1:nDatasets
    if d < 10 
        eegFilename = sprintf('simEEG_%s_spat0%d_dur%d', cond, d, target_duration);
    else
        eegFilename = sprintf('simEEG_%s_spat%d_dur%d', cond, d, target_duration);
    end
    
    % --- Load the Ground Truth Spatial Mixing Matrix ---
    key_filename = sprintf('%s_key.mat', eegFilename);
    key_filepath = fullfile(data_dir, key_filename);
    
    if ~exist(key_filepath, 'file')
        warning('Cannot find key file %s. Skipping dataset %d.', key_filename, d);
        continue;
    end
    
    key_data = load(key_filepath);
    if isfield(key_data, 'spatial_comps')
        W_true = key_data.spatial_comps;
    elseif isfield(key_data, 'W')
        W_true = key_data.W;
    else
        error('Dataset %d: Could not find spatial mixing matrix in key file.', d);
    end
    
    % --- Load this Dataset's Individual Results File ---
    subfolderName = ['results_' eegFilename];
    ds_filename = fullfile(results_dir, subfolderName, sprintf('Results_%s.mat', eegFilename));
    
    if ~exist(ds_filename, 'file')
        continue;
    end
    
    res = load(ds_filename);
    
    % Extract the actual frequencies used in this dataset (forced to column vector)
    f_peak = res.param.f_peak(:); 
    
    % --- Extract Recovery Correlations ---
    pca_tbl = res.analysis.PCA.Comp_latent_matching_corr{ki_target};
    pca_tbl = sortrows(pca_tbl, 'h_f'); 
    pca_corr = pca_tbl.corr_value;
    
    ae_tbl = res.analysis.AE.Comp_latent_matching_corr{ki_target};
    ae_tbl = sortrows(ae_tbl, 'h_f');
    ae_corr = ae_tbl.corr_value;
    
    % --- Calculate Pairwise Metrics (Diff & Mean) ---
    pairs_PCA = calculatePairwiseSimilarities(W_true, pca_corr);
    pairs_AE  = calculatePairwiseSimilarities(W_true, ae_corr);
    
    % --- Aggregate across all datasets ---
    all_pair_overlap = [all_pair_overlap; pairs_PCA.overlap]; 
    all_diff_PCA     = [all_diff_PCA; pairs_PCA.perf_diff];
    all_diff_AE      = [all_diff_AE; pairs_AE.perf_diff];
    all_mean_PCA     = [all_mean_PCA; pairs_PCA.perf_mean]; 
    all_mean_AE      = [all_mean_AE; pairs_AE.perf_mean];
    
    % --- NEW: Aggregate Identities ---
    num_pairs_in_ds = length(pairs_PCA.overlap);
    all_dataset_id  = [all_dataset_id; repmat(d, num_pairs_in_ds, 1)];
    all_freq_A      = [all_freq_A; f_peak(pairs_PCA.idx_A)];
    all_freq_B      = [all_freq_B; f_peak(pairs_PCA.idx_B)];
end

% =========================================================================
% 3. IDENTIFY SPECIFIC POINTS TO TAG
% =========================================================================
% Find Dataset 1, Pair: 6 Hz and 18 Hz (checking both A-B and B-A order)
mask_pair1 = (all_dataset_id == 1) & ...
             ((all_freq_A == 6 & all_freq_B == 18) | (all_freq_A == 18 & all_freq_B == 6));

% Find Dataset 1, Pair: 2 Hz and 10 Hz
mask_pair2 = (all_dataset_id == 1) & ...
             ((all_freq_A == 2 & all_freq_B == 10) | (all_freq_A == 10 & all_freq_B == 2));

% =========================================================================
% 4. STATISTICAL ANALYSIS & PLOTTING
% =========================================================================
fprintf('Fitting models and plotting results...\n');

% --- FIGURE 1: ABSOLUTE DIFFERENCE (Original Funnel Plot) ---
mdl_PCA_diff = fitlm(all_pair_overlap, all_diff_PCA);
mdl_AE_diff  = fitlm(all_pair_overlap, all_diff_AE);

figure('Position', [50, 100, 1200, 500], 'Name', sprintf('Pairwise Difference (k=%d)', target_k));

% Subplot 1: PCA Difference
subplot(1,2,1);
scatter(all_pair_overlap, all_diff_PCA, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
plot(mdl_PCA_diff.Variables.x1, mdl_PCA_diff.Fitted, 'k-', 'LineWidth', 2);
% Tag the special points
scatter(all_pair_overlap(mask_pair1), all_diff_PCA(mask_pair1), 150, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
text(all_pair_overlap(mask_pair1) + 0.02, all_diff_PCA(mask_pair1), '6 & 18 Hz', 'FontWeight', 'bold', 'FontSize', 10);
scatter(all_pair_overlap(mask_pair2), all_diff_PCA(mask_pair2), 150, 'g', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
text(all_pair_overlap(mask_pair2) + 0.02, all_diff_PCA(mask_pair2), '2 & 10 Hz', 'FontWeight', 'bold', 'FontSize', 10);

xlabel('Pairwise Spatial Overlap (Cosine Similarity)');
ylabel('Absolute Difference in Recovery (\Delta Corr)');
title('PCA: Performance Convergence');
ylim([-0.05 1]); xlim([0 1.05]); grid on;
txt_pca = sprintf('R^2 = %.3f\np = %.3e', mdl_PCA_diff.Rsquared.Ordinary, mdl_PCA_diff.Coefficients.pValue(2));
text(0.65, 0.85, txt_pca, 'FontSize', 12, 'BackgroundColor', 'w', 'EdgeColor', 'k');

% Subplot 2: Autoencoder Difference
subplot(1,2,2);
scatter(all_pair_overlap, all_diff_AE, 50, 'r', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
plot(mdl_AE_diff.Variables.x1, mdl_AE_diff.Fitted, 'k-', 'LineWidth', 2);
% Tag the special points
scatter(all_pair_overlap(mask_pair1), all_diff_AE(mask_pair1), 150, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
text(all_pair_overlap(mask_pair1) + 0.02, all_diff_AE(mask_pair1), '6 & 18 Hz', 'FontWeight', 'bold', 'FontSize', 10);
scatter(all_pair_overlap(mask_pair2), all_diff_AE(mask_pair2), 150, 'g', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
text(all_pair_overlap(mask_pair2) + 0.02, all_diff_AE(mask_pair2), '2 & 10 Hz', 'FontWeight', 'bold', 'FontSize', 10);

xlabel('Pairwise Spatial Overlap (Cosine Similarity)');
ylabel('Absolute Difference in Recovery (\Delta Corr)');
title('Autoencoder: Performance Convergence');
ylim([-0.05 1]); xlim([0 1.05]); grid on;
txt_ae = sprintf('R^2 = %.3f\np = %.3e', mdl_AE_diff.Rsquared.Ordinary, mdl_AE_diff.Coefficients.pValue(2));
text(0.65, 0.85, txt_ae, 'FontSize', 12, 'BackgroundColor', 'w', 'EdgeColor', 'k');


% --- FIGURE 2: MEAN PERFORMANCE ---
mdl_PCA_mean = fitlm(all_pair_overlap, all_mean_PCA);
mdl_AE_mean  = fitlm(all_pair_overlap, all_mean_AE);

figure('Position', [100, 650, 1200, 500], 'Name', sprintf('Pairwise Mean Recovery (k=%d)', target_k));

% Subplot 1: PCA Mean
subplot(1,2,1);
scatter(all_pair_overlap, all_mean_PCA, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
plot(mdl_PCA_mean.Variables.x1, mdl_PCA_mean.Fitted, 'k-', 'LineWidth', 2);
% Tag the special points
scatter(all_pair_overlap(mask_pair1), all_mean_PCA(mask_pair1), 150, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
text(all_pair_overlap(mask_pair1) + 0.02, all_mean_PCA(mask_pair1), '6 & 18 Hz', 'FontWeight', 'bold', 'FontSize', 10);
scatter(all_pair_overlap(mask_pair2), all_mean_PCA(mask_pair2), 150, 'g', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
text(all_pair_overlap(mask_pair2) + 0.02, all_mean_PCA(mask_pair2), '2 & 10 Hz', 'FontWeight', 'bold', 'FontSize', 10);

xlabel('Pairwise Spatial Overlap (Cosine Similarity)');
ylabel('Mean Recovery of Pair (Corr)');
title('PCA: Spatial Overlap Degrades Performance');
ylim([0 1.05]); xlim([0 1.05]); grid on;
txt_pca_m = sprintf('R^2 = %.3f\np = %.3e', mdl_PCA_mean.Rsquared.Ordinary, mdl_PCA_mean.Coefficients.pValue(2));
text(0.05, 0.15, txt_pca_m, 'FontSize', 12, 'BackgroundColor', 'w', 'EdgeColor', 'k');

% Subplot 2: Autoencoder Mean
subplot(1,2,2);
scatter(all_pair_overlap, all_mean_AE, 50, 'r', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
plot(mdl_AE_mean.Variables.x1, mdl_AE_mean.Fitted, 'k-', 'LineWidth', 2);
% Tag the special points
scatter(all_pair_overlap(mask_pair1), all_mean_AE(mask_pair1), 150, 'y', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
text(all_pair_overlap(mask_pair1) + 0.02, all_mean_AE(mask_pair1), '6 & 18 Hz', 'FontWeight', 'bold', 'FontSize', 10);
scatter(all_pair_overlap(mask_pair2), all_mean_AE(mask_pair2), 150, 'g', 'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
text(all_pair_overlap(mask_pair2) + 0.02, all_mean_AE(mask_pair2), '2 & 10 Hz', 'FontWeight', 'bold', 'FontSize', 10);

xlabel('Pairwise Spatial Overlap (Cosine Similarity)');
ylabel('Mean Recovery of Pair (Corr)');
title('Autoencoder: Spatial Overlap Degrades Performance');
ylim([0 1.05]); xlim([0 1.05]); grid on;
txt_ae_m = sprintf('R^2 = %.3f\np = %.3e', mdl_AE_mean.Rsquared.Ordinary, mdl_AE_mean.Coefficients.pValue(2));
text(0.05, 0.15, txt_ae_m, 'FontSize', 12, 'BackgroundColor', 'w', 'EdgeColor', 'k');