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

icasig_test = EEG.icaweights * EEG.icasphere * s_test;
icasig_test = real(icasig_test)';  % time × ICs

% Reconstruction error per latent on train & test
rec_err = zeros(num_comps, param.N_F, 2); % 3rd dim: 1=train,2=test
recon_corr_ica = zeros(num_comps, param.N_F);

for idx = 1:num_comps
    for f = 1:param.N_F
        % fit on train
       % Reconstruct latent field f from first 'idx' ICs using least-squares
        x_train = lsqlin(icasig(:,1:idx), h_f(:,f));
        h_f_reconstructed_ica = icasig(:,1:idx) * x_train;
        rec_err(idx,f,1) = mean((h_f(:,f) - h_f_reconstructed_ica).^2);
        recon_corr_ica(idx,f)= corr(h_f(:, f), h_f_reconstructed_ica);

        % test
        h_f_test_reconstructed = icasig_test(:,1:idx) * x_train;
        rec_err(idx,f,2) = mean((h_f_test(:,f) - h_f_test_reconstructed).^2);
    end
end

R = corrcoef([h_f, icasig]);
R_sub = R(1:4, 5:end);

% Plot
figure;
hold on;
imagesc(R_sub, [-1 1]); 
colorbar;
yticks(1:4);
xlim([0 num_comps+1]);
title('Correlation between true h_f and recovered ICs');
xlabel('Independent Components');
ylabel('True Latent Variables');
% Add numerical annotations
for i = 1:size(R_sub,1)
    for j = 1:size(R_sub,2)
        text(j, i, sprintf('%.2f', R_sub(i,j)), ...
            'HorizontalAlignment', 'center', ...
            'Color', 'w', ...
            'FontWeight', 'bold');
    end
end
hold off;

% === Reconstruction Correlation Heatmap ===
figure; %('Position',[100,100,600,400])
imagesc(1:num_comps, 1:param.N_F, recon_corr_ica');  % transpose so rows=latent, cols=k
xticks(1:num_comps);
xlim([0 num_comps+1]);
yticks(1:param.N_F);
colormap(jet);
clim([-1 1]);
colorbar;
xlabel('Number of ICs');
ylabel('True Latent Variable f');
title('Correlation between true and reconstructed h_f');

% Overlay numeric values
for f = 1:param.N_F
    for k = 1:num_comps
        text(k, f, sprintf('%.2f', recon_corr_ica(k,f)), ...
            'HorizontalAlignment','center', ...
            'Color','w','FontSize',8,'FontWeight','bold');
    end
end

figure;
hold on;
colors = lines(param.N_F);
for f = 1:param.N_F
    plot(1:num_comps, rec_err(:,f,1), '-', 'Marker', 'o','Color',colors(f,:), 'DisplayName',['Train Latent ',num2str(f)]);
    plot(1:num_comps, rec_err(:,f,2), '--','Marker', 'o','Color',colors(f,:), 'DisplayName',['Test  Latent ',num2str(f)]);
end
xlabel('Number of ICs'); ylabel('MSE');
title('ICA Reconstruction Error'); legend('show'); grid on;
end