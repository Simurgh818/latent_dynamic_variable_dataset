function [R2_test, MSE_test, outPCA] = runPCAAnalysis(eeg_train, eeg_test, h_train, h_test, param, num_sig_components, fs_new, results_dir)
% runPCAAnalysis Performs PCA, reconstruction, and extensive performance analysis
%
% Inputs:
%   eeg_train, eeg_test : Neural data (Neurons x Time)
%   h_train, h_test     : True latent fields (Time x F)
%   param               : Structure with fields (e.g., .N_F, .f_peak)
%   num_sig_components  : Number of components to use for detailed reconstruction plots
%   fs_new              : Sampling frequency (Hz)
%   results_dir         : Directory string to save results
%
% Outputs:
%   R2_test, MSE_test   : Vectors of error metrics per component count
%   outPCA              : Structure with detailed results

%% 1. Setup and Directory
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
h_f_colors = lines(param.N_F); % Generate colors for plots

%% 2. Run PCA
[coeff, score, ~, ~, explained] = pca(eeg_train');

% Project test data
score_test = (eeg_test' - mean(eeg_train', 1)) * coeff;

% Compute variance explained on test set
var_test = var(score_test);
explained_test = 100 * var_test / sum(var_test);
cum_explained_test = cumsum(explained_test);

%% 3. Compute R^2 and MSE for Increasing Components (1 to num_sig_components)
max_comp_check = num_sig_components; 
reconstruction_error_pca = zeros(max_comp_check, param.N_F, 2); % dim 3: 1=R2, 2=MSE (Train)
reconstruction_error_test_pca = zeros(max_comp_check, param.N_F, 2); % dim 3: 1=R2, 2=MSE (Test)

% Pre-allocate output vectors for the function return
MSE_test = zeros(max_comp_check, 1); 
R2_test_global = zeros(max_comp_check, 1); % Overall R2

for k = 1:max_comp_check
    % Train regression weights
    W = score(:, 1:k) \ h_train;
    
    % Reconstruct
    h_recon_train = score(:, 1:k) * W;
    h_recon_test  = score_test(:, 1:k) * W;
    
    % Per-feature metrics (Train)
    for f = 1:param.N_F
        % R2
        res_var = sum((h_train(:,f) - h_recon_train(:,f)).^2);
        tot_var = sum((h_train(:,f) - mean(h_train(:,f))).^2);
        reconstruction_error_pca(k, f, 1) = 1 - (res_var / tot_var);
        
        % MSE
        reconstruction_error_pca(k, f, 2) = mean((h_train(:,f) - h_recon_train(:,f)).^2);
    end

    % Per-feature metrics (Test)
    for f = 1:param.N_F
        res_var_t = sum((h_test(:,f) - h_recon_test(:,f)).^2);
        tot_var_t = sum((h_test(:,f) - mean(h_test(:,f))).^2);
        reconstruction_error_test_pca(k, f, 1) = 1 - (res_var_t / tot_var_t);
        reconstruction_error_test_pca(k, f, 2) = mean((h_test(:,f) - h_recon_test(:,f)).^2);
    end
    
    % Global Test Metrics (for output)
    MSE_test(k) = mean((h_test(:) - h_recon_test(:)).^2);
    R2_test_global(k) = 1 - sum((h_test(:) - h_recon_test(:)).^2) / sum((h_test(:) - mean(h_test(:))).^2);
end

% Final output variable for R2
R2_test = R2_test_global; 

%% 4. Detailed Reconstruction (Using specifically 'num_sig_components')
W_final = score(:, 1:num_sig_components) \ h_train;
h_recon_final = score(:, 1:num_sig_components) * W_final; % Training reconstruction

% --- Zero-Lag Correlation ---
maxLag = 200;
lags = -maxLag:maxLag;
zeroLagCorr_pca = zeros(1, param.N_F);
for f = 1:param.N_F
    c = xcorr(h_train(:,f), h_recon_final(:,f), maxLag, 'coeff');
    zeroLagCorr_pca(f) = c(lags == 0);
end

%% 5. Frequency Analysis (FFT Calculation)
N = size(h_train, 1);
trial_dur = 1; % seconds per trial
L = round(trial_dur * fs_new);
nTrials = floor(N/L);
f_freq = (0:L-1)*(fs_new/L);
nHz = L/2 + 1;
f_plot = f_freq(1:nHz);

Ht = zeros(L, param.N_F, nTrials);
Hr = zeros(L, param.N_F, nTrials);
R2_trials = zeros(L, param.N_F, nTrials);

for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);
    Z_true_sub = h_train(idx, :);
    Z_recon_sub = h_recon_final(idx, :);
    
    Ht(:,:,tr) = fft(Z_true_sub);
    Hr(:,:,tr) = fft(Z_recon_sub);
    
    for fidx = 1:param.N_F
        num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
        den = abs(Ht(:,fidx,tr)).^2 + eps;
        R2_trials(:,fidx,tr) = 1 - num./den;
    end
end

Ht_avg = mean(Ht, 3);
Hr_avg = mean(Hr, 3);
R2_avg = mean(R2_trials, 3);

% Band Averaging
bands = struct('delta', [1 4], 'theta', [4 8], 'alpha', [8 13], 'beta', [13 30], 'gamma', [30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);
band_avg_R2 = zeros(nBands, param.N_F);

for b = 1:nBands
    f_range = bands.(band_names{b});
    idx_b = f_freq >= f_range(1) & f_freq <= f_range(2);
    for fidx = 1:param.N_F
        band_avg_R2(b, fidx) = mean(R2_avg(idx_b, fidx));
    end
end


%% ============================================================
%% PLOTTING SECTION
%% ============================================================

%% Plot 1: Latent Variables Z(t) vs Reconstruction
figure('Position',[50 50 1200 150*size(h_train,2)]);
tiledlayout(size(h_train,2), 1, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('Latent variables Z(t) and PCA $\hat{z}(t)$ reconstruction.', 'Interpreter', 'latex');

for f = 1:size(h_train,2)
    nexttile; hold on;
    set(gca, 'XColor', 'none', 'YColor', 'none'); box on;
    plot(h_train(:, f),'LineStyle', '-', 'Color', h_f_colors(f, :), 'DisplayName', [sprintf('Z_{%d}', param.f_peak(f)) ' (true)']);
    plot(h_recon_final(:, f), 'LineStyle', '--','LineWidth',1,'Color', 'k', 'DisplayName', [sprintf('Z_{%d}', param.f_peak(f)) ' (recon)']);
    ylabel('amplitude');
    xlim([0 fs_new*2]); % First 2 seconds
    legend('Show','Location','eastoutside');
    
    rho = zeroLagCorr_pca(f);
    text(0.02 * fs_new, 0.7 * max(h_train(:,f)), ...
        sprintf('\\rho(0)=%.2f', rho), ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
        'Margin', 3, 'EdgeColor','k');
    hold off;
end

% Scale Bars
hold on;
x0 = 0; y0 = min(ylim)+0.2;
line([x0 x0+(fs_new)], [y0 y0], 'Color', 'k', 'LineWidth', 2, 'HandleVisibility', 'off');
text(x0+fs_new, y0-0.1, '1 sec', 'VerticalAlignment','top');
line([x0 x0], [y0 y0+2], 'Color', 'k', 'LineWidth', 2, 'HandleVisibility', 'off');
text(x0-5, y0+4, '2 a.u.', 'VerticalAlignment','bottom','HorizontalAlignment','right','Rotation',90);
hold off;
set(findall(gcf,'-property','FontSize'),'FontSize',16);
saveas(gcf, fullfile(results_dir, 'PCA_Trace_Reconstruction.png'));


%% Plot 2: PC Traces & Variance/Error
figure('Position',[50 50 1000 (num_sig_components*250)/2]);
tiledlayout(num_sig_components, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
pc_colors = lines(num_sig_components);
sgtitle('PC Traces');
for pc=1:num_sig_components
    nexttile;
    plot(score(:,pc), 'LineStyle', '-', 'Color', pc_colors(pc,:),'DisplayName', ['PC(t) ' num2str(pc)]);
    xlabel('Time bins'); ylabel('PC amplitude');
    xlim([0 1000]);
    legend('show');
end
saveas(gcf, fullfile(results_dir, 'PCA_PC_Traces.png'));


%% Plot 3: Variance and R2/MSE Curves
figure('Position',[50 50 800 600]);
tiledlayout(2, 1, 'Padding', 'compact');
nexttile;
plot(cumsum(explained), 'o-');
xline(num_sig_components, '--r', sprintf('nSigPCs=%d', num_sig_components));
xlabel('Number of PCs'); ylabel('Cumulative Variance (%)');
title('PCA Explained Variance');

nexttile;
hold on;
for f = 1:size(h_train,2)
    plot(reconstruction_error_pca(:,f,1), 'LineStyle', '-','Marker', 'o','Color',h_f_colors(f,:),'DisplayName', ['R^2 ' num2str(f)]);
    plot(reconstruction_error_pca(:,f,2), 'LineStyle', '--','Marker', 'o','Color',h_f_colors(f,:),'DisplayName', ['MSE ' num2str(f)]);
end
xticks(1:num_sig_components);
xlabel('PC Index'); ylabel('Metric Value');
title('PCA R² (Solid) and MSE (Dashed) Values');
grid('on'); hold off;
legend('show', 'Location','southeastoutside');
set(findall(gcf,'-property','FontSize'),'FontSize',14);
saveas(gcf, fullfile(results_dir, 'PCA_Metrics_vs_Components.png'));


%% Plot 4: Frequency Analysis
figure('Position',[50 50 1000 600]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('PCA Frequency Analysis: Fourier Transform');

nexttile; % Original FFT
for fidx=1:size(h_train,2)
    idx = 1:L/2+1;
    loglog(f_plot(idx), abs(Ht_avg(idx,fidx)),'Color',h_f_colors(fidx,:),'DisplayName',['Welch(Z_{' num2str(param.f_peak(fidx)) '} (f))']);
    hold on;
end
xlabel('Frequency (Hz)'); ylabel('|Z(f)|'); title('FFT Amplitude of Original Z(f)');
xlim([1, 50]); xticks([1, 4, 8, 10, 13, 20, 30, 50]);
legend('show','Location','southeastoutside'); grid on; hold off;

nexttile; % Reconstructed FFT
for fidx=1:size(h_train,2)
    idx = 1:L/2+1;
    loglog(f_plot(idx), abs(Hr_avg(idx,fidx)), 'Color',h_f_colors(fidx,:),'DisplayName',['Welch($\hat{Z}_{' num2str(param.f_peak(fidx)) '}$ (f))']);
    hold on;
end
xlabel('Frequency (Hz)'); ylabel('|Ẑ(f)|'); title('FFT Amplitude of Reconstructed $\hat{Z}$ (f)', 'Interpreter', 'latex');
xlim([1, 50]); xticks([1, 4, 8, 10, 13, 20, 30, 50]);
legend('show', 'Interpreter', 'latex','Location','southeastoutside'); grid on; hold off;
set(findall(gcf,'-property','FontSize'),'FontSize',16);
saveas(gcf, fullfile(results_dir, 'PCA_Frequency_Analysis.png'));


%% Plot 5: Band Power Bar Chart
% Calculate Band Power
band_power_true = zeros(nBands, size(h_train,2));
band_power_recon = zeros(nBands, size(h_train,2));
band_power_true_std = zeros(nBands, size(h_train,2));
band_power_recon_std = zeros(nBands, size(h_train,2));

for b = 1:nBands
    f_range = bands.(band_names{b});
    idx = f_freq >= f_range(1) & f_freq <= f_range(2);
    for fidx = 1:size(h_train,2)
        trial_power_true = mean(abs(Ht(idx,fidx,:)).^2, 1, 'omitnan');
        trial_power_recon = mean(abs(Hr(idx,fidx,:)).^2, 1, 'omitnan');
        
        band_power_true(b,fidx) = mean(trial_power_true(:));
        band_power_recon(b,fidx) = mean(trial_power_recon(:));
        band_power_true_std(b,fidx) = std(trial_power_true(:));
        band_power_recon_std(b,fidx) = std(trial_power_recon(:));
    end
end

figure('Position',[100 100 1200 250*floor(param.N_F/2)]);
tiledlayout(floor(param.N_F/2), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('Band Power Comparison: True vs Reconstructed (Mean ± SD)');
for fidx = 1:size(h_train,2)
    nexttile;
    bar_data = [band_power_true(:,fidx), band_power_recon(:,fidx)];
    bar_std = [band_power_true_std(:,fidx), band_power_recon_std(:,fidx)];
    bh = bar(bar_data); hold on;
    
    % Error bars
    ngroups = size(bar_data,1); nbars = size(bar_data,2);
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for i = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(x, bar_data(:,i), bar_std(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1);
    end
    bh(1).FaceColor = [0.3 0.6 0.9]; bh(2).FaceColor = [0.9 0.4 0.4];
    ylabel('Power (a.u.)');
    set(gca, 'XTickLabel', band_names, 'XTickLabelRotation', 45);
    title(['Z_{' num2str(param.f_peak(fidx)) '}']);
    grid on;
end
legend({'True','Reconstructed'}, 'Location','northoutside','Orientation','horizontal');
set(findall(gcf,'-property','FontSize'),'FontSize',16);
saveas(gcf, fullfile(results_dir, 'PCA_Band_Power.png'));


%% Plot 6: Band R2 Bar Chart
figure('Position',[50 50 1000 300]);
bar(band_avg_R2');
set(gca, 'XTickLabel', arrayfun(@(i) sprintf('Z_{%d}', param.f_peak(i)), 1:size(h_train,2), 'UniformOutput', false));
ylim([-1 1]);
legend(band_names, 'Location', 'southeastoutside');
ylabel('Mean R^2 in Band'); xlabel('Latent Variable');
title('PCA Band-wise Average R^2(f) per Latent Variable');
grid on;
set(findall(gcf,'-property','FontSize'),'FontSize',16);
saveas(gcf, fullfile(results_dir, 'PCA_Bandwise_R2.png'));


%% 6. Final Output Structure
outPCA = struct();
outPCA.coeff = coeff;
outPCA.score = score;
outPCA.explained = explained;
outPCA.score_test = score_test;
outPCA.reconCorr = recon_corr; % from best component model
outPCA.zeroLagCorr = zeroLagCorr_pca;
outPCA.errorTrainCurve = reconstruction_error_pca;
outPCA.errorTestCurve = reconstruction_error_test_pca;
outPCA.h_recon_train = h_recon_final;
outPCA.results_dir = results_dir;

end