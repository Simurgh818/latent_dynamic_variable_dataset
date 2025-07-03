function [coeff, score, explained, num_sig_components, reconstruction_error_pca, reconstruction_error_test_pca] = ...
    runPCAAnalysis(s_train, s_test, h_f, h_f_test, param)
% Performs PCA, MP thresholding, projection, and reconstruction error analysis
% Inputs:
%   s_train, s_test - Neuron × Time matrices (spike data)
%   h_f, h_f_test   - Time × LatentField matrices (true latents)
%   param           - Parameter struct with N_F (number of latent fields)
% Outputs:
%   coeff           - PCA components (neurons × PCs)
%   score           - Projected train data (time × PCs)
%   explained       - Variance explained by PCs
%   num_sig_components - # of significant PCs by Marchenko–Pastur

% Run PCA on training data
[coeff, score, ~, ~, explained] = pca(s_train');

% Compute eigenvalues and MP threshold
eig_vals = eig(cov(s_train'));
[N, T] = size(s_train);
Q = T / N;
sigma2 = mean(eig_vals);
lambda_max = sigma2 * (1 + sqrt(1/Q))^2;
num_sig_components = sum(eig_vals > lambda_max);

fprintf('PCA MP threshold suggests keeping %d components.\n', num_sig_components);

% Project test data into PC space
score_test = (s_test' - mean(s_train',1)) * coeff;

% Compute variance explained on test set
var_test = var(score_test);
explained_test = 100 * var_test / sum(var_test);
cum_explained_test = cumsum(explained_test);

% Plot cumulative variance
figure;
plot(cumsum(explained), 'b-', 'LineWidth', 1.5); hold on;
xline(num_sig_components, '--r', ['Significant MP PCs ' num2str(num_sig_components)],'LineWidth', 1.2);
plot(cum_explained_test, 'g--', 'LineWidth', 1.5);
title('PCA: Cumulative Variance Explained');
xlabel('PC Index'); ylabel('Cumulative Variance (%)');
ylim([0 100]);
legend('Train', 'MP Threshold', 'Test');
grid on;

% === Reconstruction Error ===
reconstruction_error_pca = zeros(num_sig_components, param.N_F);
reconstruction_error_test_pca = zeros(num_sig_components, param.N_F);

for idx = 1:num_sig_components
    for f = 1:param.N_F
        % Train
        w_train = lsqlin(score(:,1:idx), h_f(:,f));
        h_f_recon_train = score(:,1:idx) * w_train;
        reconstruction_error_pca(idx, f) = mean((h_f(:,f) - h_f_recon_train).^2);

        % Test
        h_f_recon_test = score_test(:,1:idx) * w_train;
        reconstruction_error_test_pca(idx, f) = mean((h_f_test(:,f) - h_f_recon_test).^2);
    end
end

% Plot Reconstruction Error
figure('Position', [100, 100, 600, 300]);
tiledlayout(1,1);
nexttile;
hold on;
colors = lines(param.N_F);
for f = 1:param.N_F
    plot(1:num_sig_components, reconstruction_error_pca(:, f), ...
        'Color', colors(f,:), 'DisplayName', ['Train PCA - Latent ' num2str(f)]);
    plot(1:num_sig_components, reconstruction_error_test_pca(:, f), ...
        'LineStyle', '--', 'Color', colors(f,:), 'DisplayName', ['Test PCA - Latent ' num2str(f)]);
end
xlabel('Number of PCs');
ylabel('Mean squared reconstruction error');
title('PCA Reconstruction Error for Each Latent Variable');
legend('show');
grid on;
hold off;

end