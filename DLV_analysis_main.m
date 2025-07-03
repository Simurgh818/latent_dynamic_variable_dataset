function DLV_analysis_main(s_i, param)
% Modular functions for PCA, ICA, and UMAP analyses of DLV model data
% Main wrapper to run PCA, ICA, and UMAP analyses
% Inputs:
%   s_i   : N_neur x T binary spike matrix
%   param : struct with fields (N_neur, N_F, ... )

[s_i, param, h_f] = sampleMorrellModel(param);
% Convert to double
s_train = double(s_i);

% Generate test dataset with same parameters
[s_i_test, param_test, h_f_test] = sampleMorrellModel(param);
s_test = double(s_i_test);

% 1. PCA Analysis (train & test)
[coeff, score, explained, num_sig_components, rec_err_pca] = runPCAAnalysis(s_train, s_test, h_f, h_f_test, param);

% 2. ICA Analysis (train & test)
[icasig, rec_err_ica] = runICAAnalysis(s_train, s_test, h_f, h_f_test, param, num_sig_components);

% 3. UMAP Analysis (train & test clusters)
runUMAPAnalysis(s_train, s_test, param, h_f, h_f_test, num_sig_components);

end

%% UMAP Analysis
function runUMAPAnalysis(s_train, s_test, param, h_f, h_f_test, num_sig_components)
% Runs UMAP on training & test data and computes reconstruction error
% Inputs:
%   s_train, s_test      - Neuron × Time matrices (binary spikes)
%   h_f, h_f_test        - Time × LatentField matrices (ground truth latents)
%   num_sig_components   - Number of components to use in UMAP embedding

% Convert to double and transpose to Time × Neurons
s_i_double       = double(s_train)';
s_i_test_double  = double(s_test')';

% Define UMAP hyperparameters
n_neighbors = 50;
min_dist = 0.75;
n_components = num_sig_components;

% Run UMAP on training data
[umap_s_i, umap_params] = run_umap(s_i_double, ...
    'n_neighbors', n_neighbors, ...
    'min_dist', min_dist, ...
    'n_components', n_components, ...
    'verbose', 'none', ...
    'gui', false);

% Run UMAP on test data
[umap_s_i_test, umap_params_test] = run_umap(s_i_test_double, ...
    'n_neighbors', n_neighbors, ...
    'min_dist', min_dist, ...
    'n_components', n_components, ...
    'verbose', 'none', ...
    'gui', false);

% Visualize training UMAP colored by latent structure
cluster_idx = kmeans(h_f, param.N_F);
figure('Position', [100, 100, 600, 600]);
gscatter(umap_s_i(:,1), umap_s_i(:,2), cluster_idx);
xlabel('UMAP Dimension 1');
ylabel('UMAP Dimension 2');
title({['UMAP Colored by Latent Variable h\_f'], ...
       ['n=' num2str(n_neighbors)], ['minDist=' num2str(min_dist)]});
colormap turbo;
colorbar;
grid on;

% === Reconstruction Error ===
reconstruction_error_umap = zeros(n_components, param.N_F);
reconstruction_error_test_umap = zeros(n_components, param.N_F);

for idx = 1:n_components
    for f = 1:param.N_F
        % Train reconstruction
        x_umap = lsqlin(umap_s_i(:, 1:idx), h_f(:,f));
        h_f_reconstructed_umap = umap_s_i(:, 1:idx) * x_umap;
        reconstruction_error_umap(idx,f) = mean((h_f(:,f) - h_f_reconstructed_umap).^2);

        % Test reconstruction using same weights
        h_f_test_reconstructed_umap = umap_s_i_test(:, 1:idx) * x_umap;
        reconstruction_error_test_umap(idx,f) = mean((h_f_test(:,f) - h_f_test_reconstructed_umap).^2);
    end
end

% === Plot Reconstruction Error ===
figure('Position',[100,100, 600, 300])
tiledlayout(1,1);
nexttile;
hold on;
colors = lines(param.N_F);
for f = 1:param.N_F
    plot(1:n_components, reconstruction_error_umap(:, f), ...
        'Color', colors(f,:), 'DisplayName', ['Train UMAP - Latent ' num2str(f)]);
    plot(1:n_components, reconstruction_error_test_umap(:, f), ...
        'LineStyle', '--', 'Color', colors(f,:), 'DisplayName', ['Test UMAP - Latent ' num2str(f)]);
end
xlim([1 n_components]);
xlabel('Number of UMAP components');
ylabel('Mean squared reconstruction error');
title('UMAP Reconstruction Error for Each Latent Variable');
legend('show');
grid on;
hold off;

end
