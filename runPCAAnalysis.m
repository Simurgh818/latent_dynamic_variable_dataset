function [R2_test, MSE_test, outPCA] = runPCAAnalysis(eeg_train, eeg_test, h_train, h_test, param, num_sig_components, method_dir)
% runPCAAnalysis Performs PCA, reconstruction, and extensive performance analysis
%
% Inputs:
%   eeg_train, eeg_test : Neural data (Neurons x Time)
%   h_train, h_test     : True latent fields (Time x F)
%   param               : Structure with fields (e.g., .N_F, .f_peak)
%   num_sig_components  : Number of components to use for detailed reconstruction plots
%   results_dir         : Directory string to save results
%
% Outputs:
%   R2_test, MSE_test   : Vectors of error metrics per component count
%   outPCA              : Structure with detailed results

%% 1. Setup and Directory
if ~exist(method_dir, 'dir')
    mkdir(method_dir);
end

% Define suffix for this specific run (e.g., "_k10")
file_suffix = sprintf('_k%d', num_sig_components);

h_f_colors = lines(param.N_F); 

%% 2. Run PCA
[coeff, score, ~, ~, explained] = pca(eeg_train');

% Project test data
score_test = (eeg_test' - mean(eeg_train', 1)) * coeff;

% Compute variance explained on test set
var_test = var(score_test);
explained_test = 100 * var_test / sum(var_test);

% Match components to latents
[corr_PCA, R_PCA] = match_components_to_latents(score, h_train, 'PCA');


%% 3. Compute R^2 and MSE for Increasing Components
max_comp_check = num_sig_components; 
reconstruction_error_pca = zeros(max_comp_check, param.N_F, 2); % dim 3: 1=R2, 2=MSE (Train)
reconstruction_error_test_pca = zeros(max_comp_check, param.N_F, 2); % dim 3: 1=R2, 2=MSE (Test)

MSE_test = zeros(max_comp_check, 1); 
R2_test_global = zeros(max_comp_check, 1); 

for k = 1:max_comp_check
    % Train regression weights
    W = score(:, 1:k) \ h_train;
    
    % Reconstruct
    h_recon_train = score(:, 1:k) * W;
    h_recon_test  = score_test(:, 1:k) * W;
    
    % Per-feature metrics (Train)
    for f = 1:param.N_F
        res_var = sum((h_train(:,f) - h_recon_train(:,f)).^2);
        tot_var = sum((h_train(:,f) - mean(h_train(:,f))).^2);
        reconstruction_error_pca(k, f, 1) = 1 - (res_var / tot_var);
        reconstruction_error_pca(k, f, 2) = mean((h_train(:,f) - h_recon_train(:,f)).^2);
    end

    % Per-feature metrics (Test)
    for f = 1:param.N_F
        res_var_t = sum((h_test(:,f) - h_recon_test(:,f)).^2);
        tot_var_t = sum((h_test(:,f) - mean(h_test(:,f))).^2);
        reconstruction_error_test_pca(k, f, 1) = 1 - (res_var_t / tot_var_t);
        reconstruction_error_test_pca(k, f, 2) = mean((h_test(:,f) - h_recon_test(:,f)).^2);
    end
    
    % Global Test Metrics
    MSE_test(k) = mean((h_test(:) - h_recon_test(:)).^2);
    R2_test_global(k) = 1 - sum((h_test(:) - h_recon_test(:)).^2) / sum((h_test(:) - mean(h_test(:))).^2);
end

R2_test = R2_test_global; 

%% 4. Detailed Reconstruction (Using 'num_sig_components')
W_final = score(:, 1:num_sig_components) \ h_train;
h_recon_final = score(:, 1:num_sig_components) * W_final; 

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
trial_dur = 1; 
fs_new = param.fs;
L = round(trial_dur * fs_new);
nTrials = floor(N/L);
f_freq = (0:L-1)*(fs_new/L);
nHz = L/2 + 1;
f_plot = f_freq(1:nHz);

Ht = zeros(L, param.N_F, nTrials);
Hr = zeros(L, param.N_F, nTrials);
R2_trials = zeros(L, param.N_F, nTrials);
% TODO: add comments for the code 
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

% Band Definitions
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


% ============================================================
% PLOTTING SECTION
% ============================================================
if isempty(getCurrentTask())

    % fig1 = figure('Position',[50 50 1200 150*size(h_train,2)]);
    % tiledlayout(size(h_train,2), 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    % sgtitle(['PCA (k=' num2str(num_sig_components) '): Latent variables Z(t) vs $\hat{z}$(t)'], 'Interpreter','latex');
    % 
    % for f = 1:size(h_train,2)
    %     nexttile; hold on;
    %     set(gca, 'XColor', 'none', 'YColor', 'none'); box on;
    %     plot(h_train(:, f),'LineStyle', '-', 'Color', h_f_colors(f, :), 'DisplayName', [sprintf('Z_{%s}', num2str(param.f_peak(f))) ' (true)']);
    %     plot(h_recon_final(:, f), 'LineStyle', '--','LineWidth',1,'Color', 'k', 'DisplayName', [sprintf('Z_{%s}', num2str(param.f_peak(f))) ' (recon)']);
    %     ylabel('amplitude');
    %     xlim([0 fs_new*2]); 
    %     legend('Show','Location','eastoutside');
    % 
    %     rho = zeroLagCorr_pca(f);
    %     text(0.02 * fs_new, 0.7 * max(h_train(:,f)), sprintf('\\rho(0)=%.2f', rho), ...
    %         'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor','k');
    %     hold off;
    % 
    % end
    % % scale bars (draw on last axis)
    % ax = gca;
    % hold(ax,'on');
    % x0 = 0;
    % y0 = min(ylim)+0.2;
    % line([x0 x0+param.fs], [y0 y0], 'Color', 'k', 'LineWidth', 2,'HandleVisibility', 'off');
    % text(x0+param.fs, y0-0.1, '1 sec', 'VerticalAlignment','top');
    % line([x0 x0], [y0 y0+2], 'Color', 'k', 'LineWidth', 2,'HandleVisibility', 'off');
    % text(x0-5, y0+4, '2 a.u.', 'VerticalAlignment','bottom', ...
    %     'HorizontalAlignment','right','Rotation',90);
    % set(findall(fig1,'-property','FontSize'),'FontSize',16);
    % saveas(fig1, fullfile(method_dir, ['PCA_Trace_Reconstruction' file_suffix '.png']));
    plotTimeDomainReconstruction(h_test, h_recon_test, param, 'PCA', k, zeroLagCorr_pca, method_dir);
    
    %% Plot 2: PC Traces
    if num_sig_components <= param.N_F
        num_comps_plot = num_sig_components;
    else
        num_comps_plot = param.N_F;
    end
    
    fig2 = figure('Position',[50 50 1000 (num_comps_plot*250)/2]);
    tiledlayout(num_comps_plot, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    pc_colors = lines(num_comps_plot);
    sgtitle(['PC Traces (k=' num2str(num_sig_components) ')']);
    for pc=1:num_comps_plot
        nexttile;
        plot(score(:,pc), 'LineStyle', '-', 'Color', pc_colors(pc,:),'DisplayName', ['PC(t) ' num2str(pc)]);
        xlabel('Time bins'); ylabel('PC Amp.');
        xlim([0 1000]);
        legend('show');
    end
    set(findall(fig2,'-property','FontSize'),'FontSize',16);
    saveas(fig2, fullfile(method_dir, ['PCA_PC_Traces' file_suffix '.png']));
    
    
    %% Plot 3: Metrics (Main Performance)
    fig3 = figure('Position',[50 50 800 600]);
    tiledlayout(2, 1, 'Padding', 'compact');
    nexttile;
    plot(cumsum(explained), 'o-');
    xline(num_sig_components, '--r');
    xlabel('Number of PCs'); ylabel('Cumulative Variance (%)');
    title(['PCA Explained Variance (k=' num2str(num_sig_components) ')']);
    
    nexttile;
    hold on;
    for f = 1:size(h_train,2)
        plot(reconstruction_error_pca(:,f,1), 'LineStyle', '-','Marker', 'o','Color',h_f_colors(f,:),'DisplayName', ['R^2 ' num2str(f)]);
        plot(reconstruction_error_pca(:,f,2), 'LineStyle', '--','Marker', 'o','Color',h_f_colors(f,:),'DisplayName', ['MSE ' num2str(f)]);
    end
    xticks(1:num_sig_components);
    xlabel('PC Index'); ylabel('Metric Value');
    title('PCA R² (Solid) and MSE (Dashed)');
    grid('on'); hold off;
    legend('show', 'Location','southeastoutside');
    set(findall(fig3,'-property','FontSize'),'FontSize',16);
    saveas(fig3, fullfile(method_dir, ['PCA_Metrics_vs_Components' file_suffix '.png']));
    
    
    %% Plot 4: Frequency Analysis (FFT)
    fig4 = figure('Position',[50 50 1000 600]);
    tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['PCA Frequency Analysis (k=' num2str(num_sig_components) ')']);
    
    nexttile; 
    for fidx=1:size(h_train,2)
        idx = 1:L/2+1;
        loglog(f_plot(idx), abs(Ht_avg(idx,fidx)),'Color',h_f_colors(fidx,:), ...
            'DisplayName', [sprintf('Z_{%s}(f)', num2str(param.f_peak(fidx)))]);
        hold on;
    end
    xlabel('Frequency (Hz)'); ylabel('|Z(f)|'); title('FFT Amplitude Original');
    grid on; 
    legend('show','Location','southeastoutside', 'Interpreter','latex'); hold off;
    
    nexttile; 
    for fidx=1:size(h_train,2)
        idx = 1:L/2+1;
        loglog(f_plot(idx), abs(Hr_avg(idx,fidx)), 'Color',h_f_colors(fidx,:), ...
            'DisplayName', [sprintf('\\hat{Z}_{%s}(f)', num2str(param.f_peak(fidx)))]);
        hold on;
    end
    xlabel('Frequency (Hz)'); ylabel('|Ẑ(f)|'); title('FFT Amplitude Reconstructed');
    grid on; legend('show','Location','southeastoutside', 'Interpreter','latex');
    hold off;
    
    set(findall(fig4,'-property','FontSize'),'FontSize',16);
    saveas(fig4, fullfile(method_dir, ['PCA_Frequency_Analysis' file_suffix '.png']));
    
    
    %% Plot 5: Band Power Bar Chart
    % (Simplified calculation for bar chart)
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
    
    fig5 = figure('Position',[100 100 1200 250*ceil(param.N_F/3)]);
    tiledlayout(ceil(param.N_F/3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['Band Power Comparison (k=' num2str(num_sig_components) ')']);
    for fidx = 1:size(h_train,2)
        nexttile;
        bar_data = [band_power_true(:,fidx), band_power_recon(:,fidx)];
        bar_std = [band_power_true_std(:,fidx), band_power_recon_std(:,fidx)];
        bh = bar(bar_data); hold on;
        ngroups = size(bar_data,1); nbars = size(bar_data,2);
        groupwidth = min(0.8, nbars/(nbars + 1.5));
        for i = 1:nbars
            x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
            errorbar(x, bar_data(:,i), bar_std(:,i), 'k', 'linestyle', 'none');
        end
        bh(1).FaceColor = [0.3 0.6 0.9]; bh(2).FaceColor = [0.9 0.4 0.4];
        set(gca, 'XTickLabel', band_names, 'XTickLabelRotation', 45);
        title(['Z_{' num2str(param.f_peak(fidx)) '}']);
        grid on;
    end
    legend({'True','Reconstructed'}, 'Location','northoutside','Orientation','horizontal');
    set(findall(fig5,'-property','FontSize'),'FontSize',16);
    saveas(fig5, fullfile(method_dir, ['PCA_Band_Power' file_suffix '.png']));
    
    
    %% Plot 6: Band R2 Bar Chart
    fig6 = figure('Position',[50 50 1000 300]);
    bar(band_avg_R2');
    set(gca, 'XTickLabel', arrayfun(@(i) sprintf('Z_{%s}', num2str(param.f_peak(i))), 1:size(h_train,2), 'UniformOutput', false));
    ylim([-1 1]);
    legend(band_names, 'Location', 'southeastoutside');
    title(['PCA Band-wise Average R^2 (k=' num2str(num_sig_components) ')']);
    grid on;
    set(findall(fig6,'-property','FontSize'),'FontSize',16);
    saveas(fig6, fullfile(method_dir, ['PCA_Bandwise_R2' file_suffix '.png']));
    
    
    %% Plot 7: Scatter plot: True vs Reconstructed Band Amplitudes (Mean)
    % Compute mean FFT amplitude in each band for true vs reconstructed latents
    Ht_amp = abs(Ht_avg(1:nHz, :));  
    Hr_amp = abs(Hr_avg(1:nHz, :)); 
    
    Ht_amp = Ht_amp ./ max(Ht_amp(:));
    Hr_amp = Hr_amp ./ max(Hr_amp(:));
    
    mean_band_amp_true  = zeros(nBands, size(h_train,2));
    mean_band_amp_recon = zeros(nBands, size(h_train,2));
    stdDev_band_amp_true  = zeros(nBands, size(h_train,2));
    stdDev_band_amp_recon = zeros(nBands, size(h_train,2));
    
    for b = 1:nBands
        band = band_names{b};
        f_range = bands.(band);
        idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
    
        mean_band_amp_true(b,:)  = mean(Ht_amp(idx_band,:), 1, 'omitnan');
        mean_band_amp_recon(b,:) = mean(Hr_amp(idx_band,:), 1, 'omitnan');
        
        stdDev_band_amp_true(b,:)  = std(Ht_amp(idx_band,:), 0, 1, 'omitnan');
        stdDev_band_amp_recon(b,:) = std(Hr_amp(idx_band,:), 0, 1, 'omitnan');
    end
    
    true_vals  = mean_band_amp_true(:);
    recon_vals = mean_band_amp_recon(:);
    band_labels = repelem(band_names, size(h_train,2));
    
    fig7 = figure('Position',[50 50 1600 300]);
    tiledlayout(1, param.N_F, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['True vs Reconstructed Band Mean FFT Amplitudes (per latent) (k=' num2str(num_sig_components) ')']);
    
    colors = lines(nBands);
    markers = {'o','s','d','h','^','hexagram','<','>'};
    hold on;
    
    for b = 1:nBands    
        nexttile;   
        idx_b = strcmp(band_labels, band_names{b});
        x = true_vals(idx_b);
        y = recon_vals(idx_b);
        
        for m = 1:length(markers)
            if m > size(x,1), break; end 
            hold on;
            scatter(x(m), y(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m},...
                'DisplayName', [sprintf('Z_{%s}', num2str(param.f_peak(m)))]);
            
            errorbar(x(m), y(m), stdDev_band_amp_true(b, m), stdDev_band_amp_recon(b, m), ...
                     'LineStyle', 'none', 'Color', colors(b,:), 'CapSize', 5,'HandleVisibility', 'off');
        end
    
        xfit = linspace(min(x), max(x), 100);
        plot(xfit, xfit, 'Color', colors(b,:), 'LineWidth', 2, 'DisplayName', "y=x");
    
        R_fit = corrcoef(x, y);
        R2_fit = R_fit(1,2)^2;
        text(mean(x), mean(y), sprintf('R^2=%.2f', R2_fit), 'Color', colors(b,:), 'FontSize', 12)
        if b==1
            xlabel('Mean True Amp.')
            ylabel('Mean Recon. Amp.')
        end
        title([band_names{b} ' band'])
        grid on;
    end
    nLatents = length(markers);
    proxy_handles = gobjects(nLatents + 1,1); 
    for m = 1:nLatents
        proxy_handles(m) = scatter(nan, nan, 70, 'Marker', markers{m}, ...
                                   'MarkerEdgeColor', 'k', ...
                                   'MarkerFaceColor', 'k');
    end
    proxy_handles(end) = plot(nan, nan, 'k', 'LineWidth', 2);
    legend_labels = [arrayfun(@(m) sprintf('Z_{%s}', num2str(param.f_peak(m))), 1:length(markers), 'UniformOutput', false), {'y = x'}];
    legend(proxy_handles, legend_labels, 'Location','eastoutside','TextColor','k','IconColumnWidth',7, 'NumColumns',2);
    hold off;
    set(findall(fig7,'-property','FontSize'),'FontSize',16);
    saveas(fig7, fullfile(method_dir, ['PCA_Scatter_Band_Amp_Mean' file_suffix '.png']));
    
    
    %% Plot 8: Scatter plot: True vs Reconstructed Band Amplitudes (per trial)
    
    plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, num_sig_components, "PCA", method_dir);
end
%% 6. Final Output Structure
outPCA = struct();
outPCA.coeff = coeff;
outPCA.score = score;
outPCA.explained = explained;
outPCA.score_test = score_test;
outPCA.zeroLagCorr = zeroLagCorr_pca;
outPCA.errorTrainCurve = reconstruction_error_pca;
outPCA.errorTestCurve = reconstruction_error_test_pca;
outPCA.h_recon_train = h_recon_final;
outPCA.method_dir = method_dir;
outPCA.corr_PCA = corr_PCA;
outPCA.R_full = R_PCA;

close all;
end