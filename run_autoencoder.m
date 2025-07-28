function run_autoencoder(s_eeg_like, s_eeg_like_test, numLatents, T_new, ...
    h_f_processed, h_f_processed_test, hiddenSize, epochs, lr)
    % Input data assumed in workspace: s_eeg_like, s_eeg_like_test, h_f_processed, h_f_processed_test, param, T_new

    fprintf('Running AE with learning rate = %.1e\n', lr);

    % Prepare data
    X_train = s_eeg_like;
    X_test  = s_eeg_like_test;

    % Define network
    net = feedforwardnet(hiddenSize, 'trainscg');
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'purelin';
    net.trainParam.epochs = epochs;
    net.trainParam.lr = lr;
    net.trainParam.showWindow = false;
    net.inputs{1}.processFcns = {};
    net.outputs{2}.processFcns = {};

    % Train network
    [net, ~] = train(net, X_train, X_train);

    % Compute encoder outputs
    W_enc = net.IW{1,1};
    b_enc = net.b{1};
    Z_train_c = logsig(W_enc * X_train + b_enc)';
    Z_test_c  = logsig(W_enc * X_test  + b_enc)';

    % Plot encoded features
    cols = [1.0, 0.8431, 0.0; 1.0, 0.0, 0.0; 0.0, 0.0, 1.0; 0.5, 0.0, 0.5];

    figure('Name', sprintf('AE Latents - LR=%.1e', lr), 'Position', [100, 100, 1200, 800]);
    tiledlayout(4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for f = 1:numLatents
        nexttile;
        plot(1:T_new, h_f_processed(:,f), '-',  'Color', cols(f,:), 'LineWidth', 1.5); hold on;
        plot(1:T_new, Z_train_c(:,f),      '--', 'Color', 'k', 'LineWidth', 1.5);
        ylabel(sprintf('Latent %d', f)); xlabel('Time bin'); xlim([1 2150]);
        legend({'Train', 'AE'}); grid on; hold off;

        nexttile;
        plot(1:T_new, h_f_processed_test(:,f), '-',  'Color', cols(f,:), 'LineWidth', 1.5); hold on;
        plot(1:T_new, Z_test_c(:,f),      '--', 'Color', 'k', 'LineWidth', 1.5);
        ylabel(sprintf('Latent %d', f)); xlabel('Time bin'); xlim([1 2150]);
        legend({'Test', 'AE'}); grid on; hold off;
    end
    title('True Latents (solid) vs. AE Codes (dashed)');

    % Reconstruction MSE
    numLatents = size(Z_train_c,2);
    reconErr = zeros(numLatents,1);
    reconErrTest = zeros(numLatents,1);

    for f = 1:numLatents
        w = lsqlin(Z_train_c, h_f_processed(:,f));
        w_test = lsqlin(Z_test_c, h_f_processed_test(:,f));
        reconErr(f) = mean((h_f_processed(:,f) - Z_train_c*w).^2);
        reconErrTest(f) = mean((h_f_processed_test(:,f) - Z_test_c*w_test).^2);
    end

    figure; hold on;
    c = lines(numLatents);
    for f=1:numLatents
        plot(f, reconErr(f),     'o','Color',c(f,:),'MarkerFaceColor','k');
        plot(f, reconErrTest(f),'s','Color',c(f,:),'MarkerFaceColor',c(f,:));
    end
    xlim([0.5 numLatents+0.5]);
    legend('Train','Test');
    xlabel('Latent #'); ylabel('MSE');
    title(sprintf('Reconstruction MSE (LR=%.1e)', lr));
    grid on;
end