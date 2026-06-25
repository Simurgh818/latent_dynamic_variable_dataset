function [outAE] = runAutoencoderAnalysis(X_train, X_test, H_train, H_test, bottleNeck, param,  results_dir)
% runAutoencoderAnalysis Trains AE + Linear Mapping and generates diagnostic plots
%
% Inputs:
%   X_train, X_test     : Neural data (Neurons x Time or Time x Neurons depending on your AE setup)
%   H_train, H_test     : True latent fields (Time x F)
%   bottleNeck          : Int, size of the bottleneck layer (k)
%   param               : Structure with fields .N_F, .f_peak
%   results_dir         : Directory to save results
%
% Outputs:
%   outAE               : Structure containing networks, traces, and metrics

%% 1. Setup and Directory
method_name = 'AE';
method_dir = fullfile(results_dir, method_name);
if ~exist(method_dir, 'dir')
    mkdir(method_dir);
end
file_suffix = sprintf('_k%d', bottleNeck);
h_f_colors = lines(param.N_F); 
fs_new = param.fs;

%% 2. Train Autoencoder (Unsupervised)
batch_size = 512; % 100
[net, info] = trainEEGAutoencoder(X_train, X_test,  ...
    'encoderLayerSizes', [64,32], ...
    'bottleneckSize', bottleNeck, ...
    'decoderLayerSizes', [32,64], ...
    'encoderActivations', {'tanh','tanh'}, ...
    'decoderActivations', {'tanh','tanh'}, ...
    'outputActivation', "none", ...
    'epochs', 500, ...
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

%% 4. Compute Performance Metrics
[H_recon_train, H_recon_test, Comp_latent_matching_corr, R_AE, direct_Component_Corr_ae, AE_R2_scores, freq_data] = ...
    computePerformanceMetrics(Z_train_c, Z_test_c, H_train, H_test, 'AE', bottleNeck, param);

% Separate Train Correlation for Plot 3
direct_Component_Corr_train_ae = zeros(1, param.N_F);
for f = 1:param.N_F
    c_train = corrcoef(H_train(:,f), H_recon_train(:,f));
    direct_Component_Corr_train_ae(f) = c_train(1,2);
end

% We also calculate avg_comp_corr_train here specifically for Plot 3 (Split Reconstruction)
avg_comp_corr_train = zeros(1, param.N_F);
for f = 1:param.N_F
    c_train = corrcoef(H_train(:,f), H_recon_train(:,f));
    avg_comp_corr_train(f) = c_train(1,2);
end

%% ============================================================
% PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if (isempty(getCurrentTask()) & bottleNeck==6)
    %% Plot 0: Training and Validation Loss Curve
    fig_loss = figure('Name', 'Autoencoder Loss Curve', 'Position', [100, 100, 800, 500], 'Visible', 'off');
    hold on;
    
    % info.TrainingLoss is recorded every iteration
    plot(info.TrainingLoss, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410], 'DisplayName', 'Training Loss');
    
    % info.ValidationLoss is recorded at validation frequencies (contains NaNs in between)
    val_idx = find(~isnan(info.ValidationLoss));
    if ~isempty(val_idx)
        plot(val_idx, info.ValidationLoss(val_idx), '-o', 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Validation Loss');
    end
    
    xlabel('Iteration', 'FontSize', 14);
    ylabel('MSE Loss', 'FontSize', 14);
    title(sprintf('AE Training Curve (k=%d)', bottleNeck), 'FontSize', 16);
    legend('Location', 'northeast', 'FontSize', 12);
    grid on;
    
    % Mark the epoch where early stopping occurred (if applicable)
    if ~isempty(val_idx)
        [min_val_loss, best_idx] = min(info.ValidationLoss(val_idx));
        best_iter = val_idx(best_idx);
        plot(best_iter, min_val_loss, 'k*', 'MarkerSize', 10, 'DisplayName', 'Best Model / Early Stop');
    end
    
    saveas(fig_loss, fullfile(method_dir, ['AE_Loss_Curve' file_suffix '.png']));
    close(fig_loss);
    %% Plot 1 & 2: Component Traces
    plotCTraces(bottleNeck, param, H_recon_test, method_dir, file_suffix);
    
    %% Plot 3: Detailed Reconstruction (Train & Test Split)
    fig3 = figure('Name','True vs. AE Latents', 'Position', [50, 100, 1500, 150*param.N_F], 'Visible', 'off');
    tiledlayout(param.N_F, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['AE (k=' num2str(bottleNeck) ') True Latents (solid) vs. Reconstructed (dashed)'],'FontSize',30);
    
    vis_len_train = min(fs_new*2, size(H_train,1)); 
    vis_len_test = min(fs_new*2, size(H_test,1));
    
    for f = 1:param.N_F
        % --- Train Column ---
        nexttile; hold on;
        set(gca, 'XColor', 'none', 'YColor', 'none'); box on
        plot(H_train(1:vis_len_train, f), '-',  'Color', h_f_colors(f,:), 'LineWidth', 1.5,'DisplayName', ['$Z_{' num2str(param.f_peak(f)) '}$']); 
        plot(H_recon_train(1:vis_len_train, f), '--', 'Color', 'k', 'LineWidth', 1.5,'DisplayName', ['$\hat{Z}_{' num2str(param.f_peak(f)) '}$']);
        xlim([0 fs_new]);
        if f==1, title('Training Set'); end
        legend('location', 'southeastoutside', 'Interpreter', 'latex');
        rho_train = direct_Component_Corr_train_ae(f);
        text(0.02 * fs_new, 0.05 * max(H_train(:,f)), ...
        sprintf('\\rho=%.2f', rho_train), ...
        'FontSize', 12, 'FontWeight', 'bold',...
        'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
        'Margin', 3, 'EdgeColor','k');
    
        % --- Test Column ---
        nexttile; hold on;
        set(gca, 'XColor', 'none', 'YColor', 'none'); box on
        plot(H_test(1:vis_len_test, f), '-',  'Color', h_f_colors(f,:), 'LineWidth', 1.5, 'DisplayName',[' $Z_{' num2str(param.f_peak(f)) '}$']);  
        plot(H_recon_test(1:vis_len_test, f), '--', 'Color', 'k', 'LineWidth', 1.5, 'DisplayName', ['$\hat{Z}_{' num2str(param.f_peak(f)) '}$']);
        xlim([0 fs_new]);    
        rho_test = direct_Component_Corr_ae(f);
        text(0.02 * fs_new, 0.05 * max(H_test(:,f)), ...
        sprintf('\\rho=%.2f', rho_test), ...
        'FontSize', 12, 'FontWeight', 'bold',...
        'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
        'Margin', 3, 'EdgeColor','k');
        legend('location', 'southeastoutside', 'Interpreter', 'latex');
        if f==1, title('Test Set'); end
    end
    set(findall(fig3,'-property','FontSize'),'FontSize',28);
    saveas(fig3, fullfile(method_dir, ['AE_Split_Reconstruction' file_suffix '.png']));  
    close(fig3);
    
    %% Plot 4: Frequency Analysis FFT
    save_path_fft = fullfile(method_dir, ['AE_FFT_True_vs_Recon' file_suffix '.png']);
    plotFrequencySpectra(freq_data.Ht_avg, freq_data.Hr_avg, freq_data.f_plot, 'AE', param, bottleNeck, save_path_fft);
    
    %% Plot 5: Band R2
    br2_path = fullfile(method_dir, ['AE_Bandwise_R2' file_suffix '.png']);
    plotBandwiseR2(freq_data.R2_avg, freq_data.f_axis, param, bottleNeck, 'AE', br2_path);
    
    %% Plot 6: Band Amplitude Bar Chart (Adapted for new freq_data struct)
    nBands = numel(freq_data.band_names);
    band_amp_true = zeros(nBands, param.N_F);
    band_amp_recon = zeros(nBands, param.N_F);
    band_amp_true_std = zeros(nBands, param.N_F);
    band_amp_recon_std = zeros(nBands, param.N_F);
    
    for b = 1:nBands
        for fidx = 1:param.N_F
            trial_amp_true = freq_data.true_vals{b}(fidx, :);
            trial_amp_recon = freq_data.recon_vals{b}(fidx, :);
            
            band_amp_true(b,fidx) = mean(trial_amp_true, 'omitnan');
            band_amp_recon(b,fidx) = mean(trial_amp_recon, 'omitnan');
            band_amp_true_std(b,fidx) = std(trial_amp_true, 0, 'omitnan');
            band_amp_recon_std(b,fidx) = std(trial_amp_recon, 0, 'omitnan');
        end
    end
    
    fig5 = figure('Position',[100 100 1000 250*ceil(param.N_F/3)], 'Visible', 'off');
    tiledlayout(ceil(param.N_F/3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['Band Amplitude Comparison (Mean ± SD) for k = ' num2str(bottleNeck)]);
    
    for fidx = 1:param.N_F
        nexttile;
        bar_data = [band_amp_true(:,fidx), band_amp_recon(:,fidx)];
        bar_std = [band_amp_true_std(:,fidx), band_amp_recon_std(:,fidx)];
        bh = bar(bar_data); hold on;
        
        ngroups = size(bar_data,1); nbars = size(bar_data,2);
        groupwidth = min(0.8, nbars/(nbars + 1.5));
        for i = 1:nbars
            x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
            errorbar(x, bar_data(:,i), bar_std(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1);
        end
        bh(1).FaceColor = [0.3 0.6 0.9]; bh(2).FaceColor = [0.9 0.4 0.4];
        set(gca, 'XTickLabel', freq_data.band_names, 'XTickLabelRotation', 45);
        title(['Z_{' num2str(param.f_peak(fidx)) '}']); 
        grid on;
    end
    legend({'True','Reconstructed'}, 'Location','northoutside','Orientation','horizontal');
    set(findall(fig5,'-property','FontSize'),'FontSize',16);
    saveas(fig5, fullfile(method_dir, ['AE_Band_Amplitude' file_suffix '.png']));
    close(fig5);
    
    %% Plot 7: Scatter Mean Band Amplitudes 
    fig7 = figure('Position',[50 50 1200 300], 'Visible', 'off');
    tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['AE: True vs Reconstructed Band Mean FFT Amplitudes for k= ' num2str(bottleNeck)]);
    colors = lines(nBands);
    markers = {'o','s','d','h','^','hexagram'}; 
    
    for b = 1:nBands    
        nexttile; hold on;
        
        x = mean(freq_data.true_vals{b}, 2, 'omitnan'); 
        y = mean(freq_data.recon_vals{b}, 2, 'omitnan');
        std_x = std(freq_data.true_vals{b}, 0, 2, 'omitnan');
        std_y = std(freq_data.recon_vals{b}, 0, 2, 'omitnan');
        
        for m = 1:length(markers)
            if m > numel(x), break; end
            scatter(x(m), y(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
            errorbar(x(m), y(m), std_x(m), std_y(m), 'LineStyle', 'none', 'Color', colors(b,:), 'CapSize', 5);
        end
        plot(linspace(min(x),max(x)), linspace(min(x),max(x)), 'Color', colors(b,:), 'LineWidth', 2);
        R = corrcoef(x, y);
        if numel(R) > 1, R2_val = R(1,2)^2; else, R2_val = 0; end
        text(mean(x), mean(y), sprintf('R^2=%.2f', R2_val), 'Color', colors(b,:), 'FontSize', 12);
        title([freq_data.band_names{b} ' band']); grid on;
    end
    set(findall(fig7,'-property','FontSize'),'FontSize',16);
    saveas(fig7, fullfile(method_dir, ['AE_Scatter_Mean' file_suffix '.png']));
    close(fig7);
    
    %% Plot 8: Scatter Per-Trial Band Amplitudes
    plotBandScatterPerTrial(freq_data.true_vals, freq_data.recon_vals, AE_R2_scores, freq_data.band_names, param, bottleNeck, "AE", method_dir);
end

%% 6. Outputs and Summary Saves
outAE = struct();
outAE.net = net;
outAE.info = info;
outAE.h_recon_train    = H_recon_train;
outAE.h_recon_test     = H_recon_test;
outAE.matched_R2       = freq_data.matched_R2;
outAE.results_dir      = method_dir;
outAE.Comp_latent_matching_corr = Comp_latent_matching_corr;
outAE.direct_Component_Corr  = direct_Component_Corr_ae;
outAE.Comp_latent_matching_matrix           = R_AE;
outAE.spectral_R2      = AE_R2_scores; 
% close all;

end