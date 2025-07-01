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
[coeff, score, explained, num_sig] = runPCAAnalysis(s_train, s_test,h_f, h_f_test, param);

% 2. ICA Analysis (train & test)
[icasig, rec_err_ica] = runICAAnalysis(s_train, s_test, h_f, h_f_test, param, num_sig);

% 3. UMAP Analysis (train & test clusters)
runUMAPAnalysis(s_train, s_test, param, h_f, h_f_test);

end

%% PCA Analysis
function [coeff, score, explained, num_sig_components] = runPCAAnalysis(s_train, s_test,h_f, h_f_test,param)
% Performs PCA on training, computes MP threshold, and plots for both train & test
% Outputs: coeff, score (train), explained variance, and # significant PCs

% Center & PCA
[coeff, score, ~, ~, explained] = pca(s_train');

% Eigenvalues for MP threshold
eig_vals = eig(cov(s_train'));
[N_train, T_train] = size(s_train);
Q = T_train / N_train;
sigma2 = mean(eig_vals);
lambda_max = sigma2 * (1 + sqrt(1/Q))^2;
num_sig_components = sum(eig_vals > lambda_max);
fprintf('PCA MP threshold: keep %d components.\n', num_sig_components);

% Plot cumulative variance (train)
figure; 
plot(cumsum(explained),'b-','LineWidth',1.5); hold on;
xline(num_sig_components,'--r','LineWidth',1);
title('PCA: Cumulative Variance Explained (Train)');
xlabel('PC index'); ylabel('Cumulative (%)');
legend('Train','MP Threshold');

% Project test data into PC space and plot test variance explained
score_test = (s_test' - mean(s_train',1)) * coeff;
var_test = var(score_test);
explained_test = 100 * var_test ./ sum(var_test);
cum_explained_test = cumsum(explained_test);
plot(cum_explained_test,'g--','LineWidth',1.5);
legend('Train','MP Threshold','Test');

% Reconstruction error of latent fields
rec_err_pca = zeros(num_sig_components, param.N_F, 2); % train/test
for idx = 1:num_sig_components
    for f = 1:param.N_F
        % train error: regress h_f onto top idx PCs
        w = (score(:,1:idx)' * score(:,1:idx)) \ (score(:,1:idx)' * h_f(:,f));
        rec = score(:,1:idx) * w;
        rec_err_pca(idx,f,1) = mean((h_f(:,f) - rec).^2);
        % test error
        w_test = (score_test(:,1:idx)' * score_test(:,1:idx)) \ (score_test(:,1:idx)' * h_f_test(:,f));
        rec_test = score_test(:,1:idx) * w_test;
        rec_err_pca(idx,f,2) = mean((h_f_test(:,f) - rec_test).^2);
    end
end

% Plot PCA reconstruction error
figure;
hold on;
colors = lines(param.N_F);
for f = 1:param.N_F
    plot(1:num_sig_components, rec_err_pca(:,f,1), '-', 'Color',colors(f,:), 'DisplayName',['PCA Train Latent ',num2str(f)]);
    plot(1:num_sig_components, rec_err_pca(:,f,2), '--','Color',colors(f,:), 'DisplayName',['PCA Test  Latent ',num2str(f)]);
end
xlabel('Number of PCs'); ylabel('MSE');
title('PCA Reconstruction Error'); legend('show'); grid on;

end

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
function runUMAPAnalysis(s_train, s_test, param, h_f, h_f_test)
% Runs UMAP on training & test data and colors by latent clusters

% Pre-reduce dimension for speed
s_pca_train = pca(zscore(s_train'));
s_emb_train = s_pca_train(:,1:50);

s_pca_test = pca(zscore(s_test'));
s_emb_test = s_pca_test(:,1:50);

% Number of clusters = number of latent fields
num_clusters = param.N_F;

% Cluster indices for train and test
cluster_idx_train = kmeans(h_f, num_clusters);
cluster_idx_test  = kmeans(h_f_test,  num_clusters);

% Define hyperparam grid
nN = [5 30 75]; dD = [0.5 0.75];
figure;
tiledlayout(numel(nN), numel(dD),'TileSpacing','compact');
plot_idx = 1;
for nn = nN
    for md = dD
        % Embed training data
        [emb_train, ~] = run_umap(s_emb_train, ...
            'n_neighbors', nn, 'min_dist', md, ...
            'n_components', 2, 'verbose','none','gui',false);
        nexttile(plot_idx);
        gscatter(emb_train(:,1), emb_train(:,2), cluster_idx_train);
        title(sprintf('Train: nn=%d, md=%.2f', nn, md)); axis off;
        plot_idx = plot_idx + 1;
    end
end

% Also show test embedding for last hyperparams
[emb_test, ~] = run_umap(s_emb_test, ...
    'n_neighbors', nN(end), 'min_dist', dD(end), ...
    'n_components', 2, 'verbose','none','gui',false);
figure;
gscatter(emb_test(:,1), emb_test(:,2), cluster_idx_test);
title(sprintf('Test: nn=%d, md=%.2f', nN(end), dD(end)));
axis off;
end
