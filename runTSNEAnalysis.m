function [Y_train, Y_test, rec_err_tsne, rec_err_test_tsne] = ...
    runTSNEAnalysis(s_train, s_test, param, h_f_train, h_f_test, num_dims, num_sig_components)
% runTSNEAnalysis Applies t-SNE to EEG-like data and computes reconstruction error
% 
% Inputs:
%   s_train             : Neurons × Time (EEG-like training data)
%   s_test              : Neurons × Time (EEG-like test data)
%   param               : struct containing at least param.N_F
%   h_f_train           : Time × N_F latent fields for train
%   h_f_test            : Time × N_F latent fields for test
%   num_dims            : number of t-SNE dimensions to embed into
%   num_sig_components  : number of PCs to use internally by t-SNE
%
% Outputs:
%   Y_train             : Time × num_dims t-SNE embedding of train data
%   Y_test              : Time × num_dims t-SNE embedding of test data
%   rec_err_tsne        : num_dims × N_F MSE on train latents
%   rec_err_test_tsne   : num_dims × N_F MSE on test latents

%% 1. Prepare data (Time × Features)
X_train = double(s_train)';  % Time × Neurons
X_test  = double(s_test)';   % Time × Neurons

%% 2. Run t-SNE embeddings
perplexity = 18;
Y_train = tsne(X_train, ...
    'Perplexity',         perplexity, ...
    'NumDimensions',      num_dims, ...
    'NumPCAComponents',   num_sig_components);

Y_test  = tsne(X_test, ...
    'Perplexity',         perplexity, ...
    'NumDimensions',      num_dims, ...
    'NumPCAComponents',   num_sig_components);

%% 3. Visualize train embedding colored by latent clusters
cluster_idx = kmeans(h_f_train, param.N_F);

figure('Position',[100,100,600,600]);
gscatter(Y_train(:,1), Y_train(:,2), cluster_idx);
xlabel('t-SNE Dim 1');
ylabel('t-SNE Dim 2');
title(sprintf('t-SNE (Perp=%d, D=%d)', perplexity, num_dims));
colormap turbo; colorbar; grid on;

%% 4. Compute reconstruction error of latents from t-SNE coords
rec_err_tsne      = zeros(num_dims, param.N_F);
rec_err_test_tsne = zeros(num_dims, param.N_F);

for d = 1:num_dims
    for f = 1:param.N_F
        % train regression
        w = lsqlin(Y_train(:,1:d), h_f_train(:,f));
        h_pred = Y_train(:,1:d) * w;
        rec_err_tsne(d,f) = mean((h_f_train(:,f) - h_pred).^2);
        % test regression
        h_pred_test = Y_test(:,1:d) * w;
        rec_err_test_tsne(d,f) = mean((h_f_test(:,f) - h_pred_test).^2);
    end
end

%% 5. Plot Reconstruction Error
figure('Position',[100,100,600,300]);
tiledlayout(1,1);
nexttile; hold on;
colors = lines(param.N_F);
for f = 1:param.N_F
    plot(1:num_dims, rec_err_tsne(:,f),   '-',  'Color',colors(f,:), 'DisplayName',['Train Latent ' num2str(f)]);
    plot(1:num_dims, rec_err_test_tsne(:,f),'--', 'Color',colors(f,:), 'DisplayName',['Test  Latent ' num2str(f)]);
end
xlabel('Number of t-SNE dimensions');
ylabel('Mean squared reconstruction error');
title('t-SNE Reconstruction Error per Latent');
legend('show'); grid on; hold off;

end
