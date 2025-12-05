function [R2_test, MSE_test, outAE] = runAutoencoderAnalysis(X_train, X_test, H_train, H_test, bottleNeck, param, fs_new, results_dir)
% runAutoencoderAnalysis Trains AE + Linear Mapping and generates diagnostic plots
%
% Inputs:
%   X_train, X_test     : Neural data (Neurons x Time or Time x Neurons depending on your AE setup)
%   H_train, H_test     : True latent fields (Time x F)
%   bottleNeck          : Int, size of the bottleneck layer (k)
%   param               : Structure with fields .N_F, .f_peak
%   fs_new              : Sampling frequency
%   results_dir         : Directory to save results
%
% Outputs:
%   R2_test, MSE_test   : Global performance metrics on Test set
%   outAE               : Structure containing networks, traces, and metrics

%% 1. Setup and Directory
method_name = 'Autoencoder';
method_dir = fullfile(results_dir, method_name);
if ~exist(method_dir, 'dir')
    mkdir(method_dir);
end
file_suffix = sprintf('_k%d', bottleNeck);
h_f_colors = lines(param.N_F); 

%% 2. Train Autoencoder (Unsupervised)
batch_size = 100; % 80

[net, ~] = trainEEGAutoencoder(X_train, X_test,  ...
    'encoderLayerSizes', [64,32], ...
    'bottleneckSize', bottleNeck, ...
    'decoderLayerSizes', [32,64], ...
    'encoderActivations', {'tanh','tanh'}, ...
    'decoderActivations', {'tanh','tanh'}, ...
    'outputActivation', "none", ...
    'epochs', 50, ...
    'batchSize', batch_size, ...
    'learnRate', 1e-3);

% Extract Latents
Z_train_c = double(activations(net, X_train.', 'bottleneck', 'OutputAs','rows'));
Z_test_c  = double(activations(net, X_test.',  'bottleneck', 'OutputAs','rows'));
H_train   = double(H_train);
H_test    = double(H_test);

% Ensure matching lengths
minLen = min(size(Z_train_c,1), size(H_train,1));
Z_train_c = Z_train_c(1:minLen,:);
H_train   = H_train(1:minLen,:);
minLenTest = min(size(Z_test_c,1), size(H_test,1));
Z_test_c  = Z_test_c(1:minLenTest,:);
H_test    = H_test(1:minLenTest,:);

%% 3. Linear Mapping via lsqlin
H_recon_train = zeros(size(H_train));
H_recon_test  = zeros(size(H_test));
recon_R2_train = zeros(1,param.N_F);
recon_R2_test  = zeros(1,param.N_F);

for f = 1:param.N_F
    % Linear mapping from bottleneck -> latent variable
    w = lsqlin(Z_train_c, H_train(:,f));
    
    % Reconstruct
    H_recon_train(:,f) = Z_train_c * w;
    H_recon_test(:,f)  = Z_test_c * w;
    
    % % Optional: normalize
    % H_recon_train(:,f) = H_recon_train(:,f) ./ std(H_recon_train(:,f),0,1);
    % H_recon_test(:,f)  = H_recon_test(:,f) ./ std(H_recon_test(:,f),0,1);
    
    % Component-wise R^2
    recon_R2_train(f) = 1 - sum((H_train(:,f) - H_recon_train(:,f)).^2) / sum((H_train(:,f) - mean(H_train(:,f))).^2);
    recon_R2_test(f)  = 1 - sum((H_test(:,f) - H_recon_test(:,f)).^2) / sum((H_test(:,f) - mean(H_test(:,f))).^2);
end

%% 4. Global Metrics
R2_test  = 1 - sum((H_test(:) - H_recon_test(:)).^2) / sum((H_test(:) - mean(H_test(:))).^2);
MSE_test = mean((H_test(:) - H_recon_test(:)).^2);
% 
% for f = 1:param.N_F
%     disp(corr(H_test(:,f), H_recon_test(:,f)));
% end
%% Zero-lag cross correlation
%   Z_true     [T × F]  — true latent time series
Z_true_train = H_train;
Z_true_test = H_test;
%   Z_recon_pca [T × F] — latent reconstructed via PCA
Z_recon_train = H_recon_train;
Z_recon_test = H_recon_test;

maxLag = 200;               % number of lags to compute on either side
lags   = -maxLag:maxLag;    

[~, param.N_F] = size(Z_true_train);
zeroLagCorr_train = zeros(1, param.N_F);
zeroLagCorr_test = zeros(1, param.N_F);
for f = 1:param.N_F
    % compute normalized cross‐correlation
    c_train = xcorr(Z_true_train(:,f), Z_recon_train(:,f), maxLag, 'coeff');
    c_test = xcorr(Z_true_test(:,f), Z_recon_test(:,f), maxLag, 'coeff');
    % pick out the zero‐lag value
    zeroLagCorr_train(f) = c_train(lags==0);
    zeroLagCorr_test(f) = c_test(lags==0);
end

% ============================================================
% PLOTTING SECTION
% ============================================================

% %% Plot 1: Mapping Correlation Bar Chart
% fig1 = figure('Position',[50 50 400 250]);
% bar(corr_vals);
% xlabel('Latent dim (h_f)');
% ylabel('Corr(predicted, true)');
% title(['Mapping accuracy AE (k=' num2str(bottleNeck) ')']);
% grid on;
% saveas(fig1, fullfile(method_dir, ['AE_Mapping_Corr' file_suffix '.png']));

% %% Plot 2: Trace Overlays (Simple)
% fig2 = figure('Position',[50 50 1000 700]);
% tiledlayout(param.N_F, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
% plot_len = min(1000, size(H_train,1));
% for i = 1:min(param.N_F, size(H_train,2))
%     nexttile;
%     plot(H_train(1:plot_len,i), 'b', 'LineWidth', 1.5); hold on;
%     plot(pred_H_train(1:plot_len,i), 'r--', 'LineWidth', 1.5);
%     legend('True', 'Predicted','Location','eastoutside');
%     title(sprintf('Latent %d: corr = %.2f', i, corr_vals(i)));
%     grid on;
% end
% saveas(fig2, fullfile(method_dir, ['AE_Simple_Traces' file_suffix '.png']));

%% Plot 3: Detailed Reconstruction (Train & Test Split)
fig3 = figure('Name','True vs. AE Latents', 'Position', [100, 100, 1200, 110*param.N_F]);
tiledlayout(param.N_F, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['AE (k=' num2str(bottleNeck) ') True Latents (solid) vs. Reconstructed (dashed)']);

% Limits for plots
vis_len_train = min(fs_new*2, size(H_train,1)); % 2 seconds
vis_len_test = min(fs_new*2, size(H_test,1));

for f = 1:param.N_F
    % --- Train Column ---
    nexttile; hold on;
    set(gca, 'XColor', 'none', 'YColor', 'none'); box on
    plot(H_train(1:vis_len_train, f), '-',  'Color', h_f_colors(f,:), 'LineWidth', 1.5,'DisplayName', ['$Z_{' num2str(param.f_peak(f)) '}$']); 
    plot(H_recon_train(1:vis_len_train, f), '--', 'Color', 'k', 'LineWidth', 1.5,'DisplayName', ['$\hat{Z}_{' num2str(param.f_peak(f)) '}$']);
    xlim([0 fs_new*2]);
    if f==1, title('Training Set'); end
    legend('location', 'southeastoutside', 'Interpreter', 'latex');
    rho_train = zeroLagCorr_train(f);
    text(0.02 * fs_new, 0.05 * max(H_train(:,f)), ...
    sprintf('\\rho(0)=%.2f', rho_train), ...
    'FontSize', 12, 'FontWeight', 'bold',...
    'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
    'Margin', 3, 'EdgeColor','k');
    % Scale bar on last plot
      if f == param.N_F
        %set(gca, 'XTickLabel', []);
        x0 = 0;           % starting point of scale bar (x)
        y0 = min(ylim)+0.2; % bottom of scale bar (y)
        
        % time bar (horizontal)
        line([x0 x0+(fs_new)], [y0 y0], 'Color', 'k', 'LineWidth', 2,'HandleVisibility', 'off');
        text(x0+fs_new, y0-0.1, '1 sec', 'VerticalAlignment','top');
        
        % amplitude bar (vertical)
        line([x0 x0], [y0 y0+2], 'Color', 'k', 'LineWidth', 2,'HandleVisibility', 'off');
        text(x0-5, y0+4, '2 a.u.', 'VerticalAlignment','bottom','HorizontalAlignment','right','Rotation',90);
        hold off;
     end

    % --- Test Column ---
    nexttile; hold on;
    set(gca, 'XColor', 'none', 'YColor', 'none'); box on
    plot(H_test(1:vis_len_test, f), '-',  'Color', h_f_colors(f,:), 'LineWidth', 1.5, 'DisplayName',[' $Z_{' num2str(param.f_peak(f)) '}$']);  
    plot(H_recon_test(1:vis_len_test, f), '--', 'Color', 'k', 'LineWidth', 1.5, 'DisplayName', ['$\hat{Z}_{' num2str(param.f_peak(f)) '}$']);
    xlim([0 fs_new*2]);
        rho_test = zeroLagCorr_test(f);
    text(0.02 * fs_new, 0.05 * max(H_test(:,f)), ...
    sprintf('\\rho(0)=%.2f', rho_test), ...
    'FontSize', 12, 'FontWeight', 'bold',...
    'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
    'Margin', 3, 'EdgeColor','k');
    legend('location', 'southeastoutside', 'Interpreter', 'latex');
    if f==1, title('Test Set'); end
        if f == param.N_F
        %set(gca, 'XTickLabel', []);
        x0 = 0;           % starting point of scale bar (x)
        y0 = min(ylim)+0.2; % bottom of scale bar (y)
        
        % time bar (horizontal)
        line([x0 x0+(fs_new)], [y0 y0], 'Color', 'k', 'LineWidth', 2,'HandleVisibility', 'off');
        text(x0+fs_new, y0-0.1, '1 sec', 'VerticalAlignment','top');
        
        % amplitude bar (vertical)
        line([x0 x0], [y0 y0+2], 'Color', 'k', 'LineWidth', 2,'HandleVisibility', 'off');
        text(x0-5, y0+4, '2 a.u.', 'VerticalAlignment','bottom','HorizontalAlignment','right','Rotation',90);
        hold off;
     end
end
set(findall(fig3,'-property','FontSize'),'FontSize',16);
saveas(fig3, fullfile(method_dir, ['AE_Split_Reconstruction' file_suffix '.png']));


%% 5. Frequency Analysis (FFT)
N = size(H_recon_test, 1);
trial_dur = 1;              
L         = round(trial_dur * fs_new);   
nTrials   = floor(N/L);
f_freq    = (0:L-1)*(fs_new/L);   
nHz       = L/2+1;
f_plot    = f_freq(1:nHz);

% Initialize containers
R2_trials_ae = zeros(L, param.N_F, nTrials);
Ht_ae = zeros(L, param.N_F, nTrials);
Hr_ae = zeros(L, param.N_F, nTrials);

for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);      
    Z_true_sub  = H_test(idx,:);      
    Z_recon_sub = H_recon_test(idx,:);
    
    Ht_ae(:,:,tr) = fft(Z_true_sub);   
    Hr_ae(:,:,tr) = fft(Z_recon_sub);
    
    for fidx = 1:param.N_F
        num = abs(Ht_ae(:,fidx,tr) - Hr_ae(:,fidx,tr)).^2;
        den = abs(Ht_ae(:,fidx,tr)).^2 + eps;
        R2_trials_ae(:,fidx,tr) = 1 - num./den;
    end
end

% Averages
R2_avg_ae = mean(R2_trials_ae, 3);
Ht_avg_ae = mean(Ht_ae, 3);
Hr_avg_ae = mean(Hr_ae, 3);

% Band Definitions
bands = struct('delta', [1 4], 'theta', [4 8], 'alpha', [8 13], 'beta', [13 30], 'gamma', [30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);
band_avg_R2_ae = zeros(nBands, param.N_F);

for b = 1:nBands
    f_range = bands.(band_names{b});
    idx = f_freq >= f_range(1) & f_freq <= f_range(2);
    for fidx = 1:param.N_F
        band_avg_R2_ae(b, fidx) = mean(R2_avg_ae(idx, fidx));
    end
end

%% Plot 4: FFT Analysis
fig4 = figure('Position',[50 50 1000 600]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['Autoencoder Frequency Analysis, k= ' num2str(bottleNeck)]);

nexttile
for fidx=1:param.N_F
    idx = 1:L/2+1;
    loglog(f_plot(idx), abs(Ht_avg_ae(idx,fidx)),'Color',h_f_colors(fidx,:));
    hold on;
end
xlabel('Frequency (Hz)'); ylabel('|Z(f)|'); title('FFT Amplitude Original');
xlim([1, 50]); xticks([1, 4, 8, 13, 30, 50]); grid on; hold off;

nexttile
for fidx=1:param.N_F
    idx = 1:L/2+1;
    loglog(f_plot(idx), abs(Hr_avg_ae(idx,fidx)), 'Color',h_f_colors(fidx,:));
    hold on;
end
xlabel('Frequency (Hz)'); ylabel('|Ẑ(f)|'); title('FFT Amplitude Reconstructed');
xlim([1, 50]); xticks([1, 4, 8, 13, 30, 50]); grid on; hold off;
set(findall(fig4,'-property','FontSize'),'FontSize',16);
saveas(fig4, fullfile(method_dir, ['AE_Frequency_Analysis' file_suffix '.png']));


%% Plot 5: Band Power Bar Chart
band_power_true = zeros(nBands, param.N_F);
band_power_recon = zeros(nBands, param.N_F);
band_power_true_std = zeros(nBands, param.N_F);
band_power_recon_std = zeros(nBands, param.N_F);

for b = 1:nBands
    f_range = bands.(band_names{b});
    idx = f_freq >= f_range(1) & f_freq <= f_range(2);
    for fidx = 1:param.N_F
        trial_power_true = mean(abs(Ht_ae(idx,fidx,:)).^2, 1, 'omitnan');
        trial_power_recon = mean(abs(Hr_ae(idx,fidx,:)).^2, 1, 'omitnan');
        
        band_power_true(b,fidx) = mean(trial_power_true(:));
        band_power_recon(b,fidx) = mean(trial_power_recon(:));
        band_power_true_std(b,fidx) = std(trial_power_true(:));
        band_power_recon_std(b,fidx) = std(trial_power_recon(:));
    end
end

fig5 = figure('Position',[100 100 1000 250*ceil(param.N_F/3)]);
tiledlayout(ceil(param.N_F/3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['Band Power Comparison (Mean ± SD) for k = ' num2str(bottleNeck)]);

for fidx = 1:param.N_F
    nexttile;
    bar_data = [band_power_true(:,fidx), band_power_recon(:,fidx)];
    bar_std = [band_power_true_std(:,fidx), band_power_recon_std(:,fidx)];
    bh = bar(bar_data); hold on;
    
    ngroups = size(bar_data,1); nbars = size(bar_data,2);
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for i = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(x, bar_data(:,i), bar_std(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1);
    end
    bh(1).FaceColor = [0.3 0.6 0.9]; bh(2).FaceColor = [0.9 0.4 0.4];
    set(gca, 'XTickLabel', band_names, 'XTickLabelRotation', 45);
    title(['Z_{' num2str(param.f_peak(fidx)) '}']); 
    grid on;
end
legend({'True','Reconstructed'}, 'Location','northoutside','Orientation','horizontal');
set(findall(fig5,'-property','FontSize'),'FontSize',16);
saveas(fig5, fullfile(method_dir, ['AE_Band_Power' file_suffix '.png']));


%% Plot 6: Band R2
fig6 = figure('Position',[50 50 1000 300]);
bar(band_avg_R2_ae');
set(gca, 'XTickLabel', arrayfun(@(i) sprintf('Z_{%d}',param.f_peak(i)), 1:param.N_F, 'UniformOutput', false));
ylim([-1 1]);
legend(band_names, 'Location', 'southeastoutside');
ylabel('Mean R^2'); xlabel('Latent Variable'); title(['Autoencoder Band-wise R^2 for k= ' num2str(bottleNeck)]);
grid on;
set(findall(fig6,'-property','FontSize'),'FontSize',16);
saveas(fig6, fullfile(method_dir, ['AE_Bandwise_R2' file_suffix '.png']));


%% Plot 7: Scatter Mean Band Amplitudes
Ht_amp = abs(Ht_avg_ae(1:nHz, :));
Hr_amp = abs(Hr_avg_ae(1:nHz, :));
Ht_amp = Ht_amp ./ max(Ht_amp(:)); % Normalize
Hr_amp = Hr_amp ./ max(Hr_amp(:));

mean_band_amp_true = zeros(nBands, param.N_F);
mean_band_amp_recon = zeros(nBands, param.N_F);
std_band_amp_true = zeros(nBands, param.N_F);
std_band_amp_recon = zeros(nBands, param.N_F);

for b = 1:nBands
    f_range = bands.(band_names{b});
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
    mean_band_amp_true(b,:)  = mean(Ht_amp(idx_band,:), 1, 'omitnan');
    mean_band_amp_recon(b,:) = mean(Hr_amp(idx_band,:), 1, 'omitnan');
    std_band_amp_true(b,:)  = std(Ht_amp(idx_band,:), 0, 1, 'omitnan');
    std_band_amp_recon(b,:) = std(Hr_amp(idx_band,:), 0, 1, 'omitnan');
end

true_vals = mean_band_amp_true(:);
recon_vals = mean_band_amp_recon(:);
band_labels = repelem(band_names, param.N_F);

fig7 = figure('Position',[50 50 1200 300]);
tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['AE: True vs Reconstructed Band Mean FFT Amplitudes for k= ' num2str(bottleNeck)]);
colors = lines(nBands);
markers = {'o','s','d','h','^','hexagram','<','>'};

for b = 1:nBands    
    nexttile;   
    idx_b = strcmp(band_labels, band_names{b});
    x = true_vals(idx_b); y = recon_vals(idx_b);
    hold on;
    for m = 1:length(markers)
        if m > numel(x), break; end
        scatter(x(m), y(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
        errorbar(x(m), y(m), std_band_amp_true(b, m), std_band_amp_recon(b, m), ...
                 'LineStyle', 'none', 'Color', colors(b,:), 'CapSize', 5);
    end
    plot(linspace(min(x),max(x)), linspace(min(x),max(x)), 'Color', colors(b,:), 'LineWidth', 2);
    R = corrcoef(x, y);
    text(mean(x), mean(y), sprintf('R^2=%.2f', R(1,2)^2), 'Color', colors(b,:), 'FontSize', 12);
    title([band_names{b} ' band']); grid on;
end
set(findall(fig7,'-property','FontSize'),'FontSize',16);
saveas(fig7, fullfile(method_dir, ['AE_Scatter_Mean' file_suffix '.png']));


%% Plot 8: Scatter Per-Trial Band Amplitudes
Ht_amp_tr = abs(Ht_ae(1:nHz, :, :)) ./ max(abs(Ht_ae(:)));
Hr_amp_tr = abs(Hr_ae(1:nHz, :, :)) ./ max(abs(Hr_ae(:)));
true_vals_band = cell(nBands, 1);
recon_vals_band = cell(nBands, 1);

for b = 1:nBands
    f_range = bands.(band_names{b});
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
    temp_t = squeeze(mean(Ht_amp_tr(idx_band, :, :), 1, 'omitnan'));
    temp_r = squeeze(mean(Hr_amp_tr(idx_band, :, :), 1, 'omitnan'));
    true_vals_band{b} = temp_t(:);
    recon_vals_band{b} = temp_r(:);
end

fig8 = figure('Position',[50 50 1200 300]);
tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['AE: True vs Reconstructed Per-Trial Band Amplitudes for k= ' num2str(bottleNeck)]);

for b = 1:nBands
    nexttile; hold on;
    x = true_vals_band{b}; y = recon_vals_band{b};
    scatter(x, y, 30, 'Marker', markers{b}, 'MarkerEdgeColor', colors(b,:), 'MarkerFaceAlpha', 0.3,...
        'DisplayName', [sprintf('Z_{%s}', band_names{b})]);
    plot(linspace(min(x),max(x)), linspace(min(x),max(x)), 'k--', 'LineWidth', 1.5, 'DisplayName', 'y=x');
    R = corrcoef(x,y);
    text(mean(x), mean(y), sprintf('R^2=%.2f', R(1,2)^2), 'Color', 'k', 'FontSize', 12);
    title(band_names{b}); grid on;
    if b==1
        xlabel('True Band Amp.')
        ylabel('Recon. Band Amp.')
    end

    legend('Location','southoutside','TextColor','k','Orientation','horizontal');
end
set(findall(fig8,'-property','FontSize'),'FontSize',16);
saveas(fig8, fullfile(method_dir, ['AE_Scatter_Trials' file_suffix '.png']));


%% 6. Outputs and Summary Saves
outAE = struct();
outAE.net = net;
outAE.H_recon_train     = H_recon_train;
outAE.H_recon_test      = H_recon_test;
outAE.R2_test          = R2_test;
outAE.MSE_test         = MSE_test;
outAE.component_R2     = recon_R2_test;
outAE.results_dir      = method_dir;

close All;

end