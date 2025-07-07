function [umap_s_i, umap_s_i_test, reconstruction_error_umap, reconstruction_error_test_umap] = ...
    runUMAPAnalysis(n_neighbors, min_dist, s_train, s_test, param, h_f_train, h_f_test, num_sig_components)
% runUMAPAnalysis applies UMAP to spike data and computes reconstruction error
% Inputs:
%   s_train         : Neurons × Time training spike data
%   s_test          : Neurons × Time test spike data
%   h_f_train       : Time × N_F latent fields (processed)
%   h_f_test        : Time × N_F latent fields (processed, test)
%   param           : Struct containing model parameters (incl. N_F)
%   num_sig_components : Number of components to evaluate
%
% Outputs:
%   umap_s_i        : Time × UMAP dims (train embedding)
%   umap_s_i_test   : Time × UMAP dims (test embedding)
%   reconstruction_error_umap       : [n_comp × N_F] MSE (train)
%   reconstruction_error_test_umap : [n_comp × N_F] MSE (test)

%% 1. Set Parameters
n_components = num_sig_components;

%% 2. Format Data
s_train_T = double(s_train)';       % Time × Neurons
s_test_T  = double(s_test)';        % Time × Neurons

%% 3. Run UMAP Embedding
[umap_s_i, ~] = run_umap(s_train_T, ...
    'n_neighbors', n_neighbors, ...
    'min_dist', min_dist, ...
    'n_components', n_components, ...
    'verbose', 'none', ...
    'gui', false);

[umap_s_i_test, ~] = run_umap(s_test_T, ...
    'n_neighbors', n_neighbors, ...
    'min_dist', min_dist, ...
    'n_components', n_components, ...
    'verbose', 'none', ...
    'gui', false);

%% 4. Visualize UMAP Projection (Train)
cluster_idx = kmeans(h_f_train, param.N_F);  % cluster based on true latent dynamics

figure('Position', [100, 100, 600, 600]);
gscatter(umap_s_i(:,1), umap_s_i(:,2), cluster_idx);
xlabel('UMAP Dimension 1');
ylabel('UMAP Dimension 2');
title({['UMAP Colored by Latent Variable h_f'], ...
       ['n = ' num2str(n_neighbors)], ...
       ['minDist = ' num2str(min_dist)]});
colormap turbo;
colorbar; grid on;

%% 5. Reconstruction Error
reconstruction_error_umap       = zeros(n_components, param.N_F);
reconstruction_error_test_umap = zeros(n_components, param.N_F);

for idx = 1:n_components
    for f = 1:param.N_F
        % Fit on training set
        weights = lsqlin(umap_s_i(:, 1:idx), h_f_train(:,f));
        h_f_pred_train = umap_s_i(:, 1:idx) * weights;
        reconstruction_error_umap(idx, f) = mean((h_f_train(:,f) - h_f_pred_train).^2);

        % Apply to test set
        h_f_pred_test = umap_s_i_test(:, 1:idx) * weights;
        reconstruction_error_test_umap(idx, f) = mean((h_f_test(:,f) - h_f_pred_test).^2);
    end
end

%% 6. Plot Reconstruction Error
figure('Position', [100, 100, 600, 300]);
tiledlayout(1, 1);
nexttile;
hold on;
colors = lines(param.N_F);
for f = 1:param.N_F
    plot(1:n_components, reconstruction_error_umap(:, f), ...
        '-', 'Color', colors(f,:), 'DisplayName', ['Train - Latent ' num2str(f)]);
    plot(1:n_components, reconstruction_error_test_umap(:, f), ...
        '--', 'Color', colors(f,:), 'DisplayName', ['Test  - Latent ' num2str(f)]);
end
xlabel('Number of UMAP components');
ylabel('Mean squared reconstruction error');
title('UMAP Reconstruction Error per Latent Variable');
legend('show');
grid on;
hold off;

end
