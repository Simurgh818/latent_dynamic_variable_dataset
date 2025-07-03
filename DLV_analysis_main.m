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
%% PCA



%% ICA Analysis
function [icasig, rec_err] = runICAAnalysis(s_train, s_test, h_f, h_f_test, param, num_comps)
% Runs EEGLAB ICA on training, then computes reconstruction error on train & test

% Prepare EEG structure for train
EEG = eeg_emptyset();
EEG.data = s_train;
EEG.nbchan = size(s_train,1); EEG.pnts = size(s_train,2);
EEG.trials = 1; EEG.srate = 100; EEG.xmin = 0;
EEG = eeg_checkset(EEG);

% Run ICA with PCA whitening to num_comps
EEG = pop_runica(EEG, 'extended',1, 'pca', num_comps, 'interrupt','off');
icasig = double(EEG.icaact)';  % time × ICs

icasig_test = EEG.icawinv' * s_test;
icasig_test = real(icasig_test)';  % time × ICs

% Reconstruction error per latent on train & test
rec_err = zeros(num_comps, param.N_F, 2); % 3rd dim: 1=train,2=test
for idx = 1:num_comps
    for f = 1:param.N_F
        % fit on train
       % Reconstruct latent field f from first 'idx' ICs using least-squares
        x_train = lsqlin(icasig(:,1:idx), h_f(:,f));
        h_f_reconstructed_ica = icasig(:,1:idx) * x_train;
        rec_err(idx,f,1) = mean((h_f(:,f) - h_f_reconstructed_ica).^2);
        % test
        h_f_test_reconstructed = icasig_test(:,1:idx) * x_train;
        rec_err(idx,f,2) = mean((h_f_test(:,f) - h_f_test_reconstructed).^2);
    end
end

% Plot
figure;
hold on;
colors = lines(param.N_F);
for f = 1:param.N_F
    plot(1:num_comps, rec_err(:,f,1), '-', 'Color',colors(f,:), 'DisplayName',['Train Latent ',num2str(f)]);
    plot(1:num_comps, rec_err(:,f,2), '--','Color',colors(f,:), 'DisplayName',['Test  Latent ',num2str(f)]);
end
xlabel('Number of ICs'); ylabel('MSE');
title('ICA Reconstruction Error'); legend('show'); grid on;
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
