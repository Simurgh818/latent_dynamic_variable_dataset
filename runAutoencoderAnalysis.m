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

%% 2. Train 2D Spectrogram Convolutional Autoencoder
batch_size = 4; 

% --- 2.1 Calculate Spectrograms (STFT) ---
disp('Extracting 2D Spectrogram Features...');
window = round(param.fs * 0.5);   % 500ms sliding window
overlap = round(param.fs * 0.4);  % 400ms overlap (step size of 100ms)
nfft = window;

% Precompute for Train data to get output dimensions
[~, F, T_train] = spectrogram(X_train(1,:), window, overlap, nfft, param.fs);
freq_idx = F <= 50; % Keep only frequencies up to 50Hz to save memory!
NumFreqs = sum(freq_idx);

X_train_spec = zeros(size(X_train,1), NumFreqs, 1, length(T_train));
for ch = 1:size(X_train,1)
    [S, ~, ~] = spectrogram(X_train(ch,:), window, overlap, nfft, param.fs);
    X_train_spec(ch, :, 1, :) = log(abs(S(freq_idx, :)) + 1); % Log-transform stabilizes CNN gradients
end

% Precompute for Test data
[~, ~, T_test] = spectrogram(X_test(1,:), window, overlap, nfft, param.fs);
X_test_spec = zeros(size(X_test,1), NumFreqs, 1, length(T_test));
for ch = 1:size(X_test,1)
    [S, ~, ~] = spectrogram(X_test(ch,:), window, overlap, nfft, param.fs);
    X_test_spec(ch, :, 1, :) = log(abs(S(freq_idx, :)) + 1);
end

% --- 2.2 Train the Network ---
ckpt_dir = tempname; 
mkdir(ckpt_dir); 

[net, info] = trainEEG_CAE(X_train_spec, X_test_spec,  ...
    'bottleneckSize', bottleNeck, ...
    'epochs', 150, ... 
    'batchSize', batch_size, ...
    'learnRate', 1e-4, ...
    'checkpointPath', ckpt_dir);

% --- 2.3 Extract Latents ---
% The CNN outputs latents corresponding to the STFT TimeWindows
Z_train_stft = double(activations(net, X_train_spec, 'bottleneck', 'OutputAs','rows'));
Z_test_stft  = double(activations(net, X_test_spec,  'bottleneck', 'OutputAs','rows'));

% --- 2.4 Interpolate Latents back to Native EEG Time Resolution ---
% Because the STFT compresses time, we must stretch Z back out so it perfectly 
% aligns with your Ground Truth H matrices for the correlation math!
orig_time_train = (0:size(H_train,1)-1) / param.fs;
orig_time_test  = (0:size(H_test,1)-1) / param.fs;

% Interpolate STFT centers (T_train) to native times (orig_time_train)
Z_train_c = interp1(T_train, Z_train_stft, orig_time_train, 'linear', 'extrap');
Z_test_c  = interp1(T_test, Z_test_stft, orig_time_test, 'linear', 'extrap');

H_train   = double(H_train);
H_test    = double(H_test);

% Ensure matching lengths
minLen = min(size(Z_train_c,1), size(H_train,1));
Z_train_c = Z_train_c(1:minLen,:);
H_train   = H_train(1:minLen,:);

minLenTest = min(size(Z_test_c,1), size(H_test,1));
Z_test_c  = Z_test_c(1:minLenTest,:);
H_test    = H_test(1:minLenTest,:);

%% --- NEW: 3.5 Evaluate Checkpoints every 5 Epochs ---
disp('Evaluating Checkpoints for Z-Z_hat Correlation...');
ckpt_files = dir(fullfile(ckpt_dir, 'net_checkpoint__*.mat'));

% Extract iterations to sort them chronologically
iters = zeros(length(ckpt_files), 1);
for i = 1:length(ckpt_files)
    parts = split(ckpt_files(i).name, '__');
    iters(i) = str2double(parts{2});
end
[sorted_iters, sort_idx] = sort(iters);
ckpt_files = ckpt_files(sort_idx);

iters_per_epoch = max(1, floor(size(X_train_spec, 4) / batch_size)); 
eval_every_n_epochs = 2;

% Create target epochs up to the point early stopping was triggered
max_epoch_reached = floor(max(sorted_iters) / iters_per_epoch);
target_epochs = eval_every_n_epochs : eval_every_n_epochs : max_epoch_reached;
target_iters = target_epochs * iters_per_epoch;

history_corr = nan(length(target_epochs), param.N_F);

for i = 1:length(target_epochs)
    % Load network state at this epoch
    [~, closest_idx] = min(abs(sorted_iters - target_iters(i)));
    ckpt_data = load(fullfile(ckpt_dir, ckpt_files(closest_idx).name));
    temp_net = ckpt_data.net; 
    
    % Extract latents using the 2D Spectrograms
    Z_train_stft_tmp = double(activations(temp_net, X_train_spec, 'bottleneck', 'OutputAs','rows'));
    Z_test_stft_tmp  = double(activations(temp_net, X_test_spec,  'bottleneck', 'OutputAs','rows'));
    
    % Interpolate back to native time
    Z_train_tmp = interp1(T_train, Z_train_stft_tmp, orig_time_train, 'linear', 'extrap');
    Z_test_tmp  = interp1(T_test, Z_test_stft_tmp, orig_time_test, 'linear', 'extrap');
    
    % Match lengths
    Z_train_tmp = Z_train_tmp(1:minLen,:);
    Z_test_tmp  = Z_test_tmp(1:minLenTest,:);
    
    % Subspace mapping 
    W_multi = pinv(Z_train_tmp) * H_train;
    Z_recon_test_tmp = Z_test_tmp * W_multi;
    
    % Calculate Pearson Correlation
    for f = 1:param.N_F
        c = corrcoef(H_test(:,f), Z_recon_test_tmp(:,f));
        history_corr(i, f) = c(1,2);
    end
end

% Clean up checkpoints to free up hard drive space!
rmdir(ckpt_dir, 's');
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
if (isempty(getCurrentTask()) & bottleNeck==10)
  %% Plot 0: Training and Validation Loss Curve
    fig_loss = figure('Name', 'Autoencoder Loss Curve', 'Position', [100, 100, 800, 500], 'Visible', 'off');
    hold on;
    
    % --- FIXED: Now uses the Spectrogram dimensions safely ---
    iters_per_epoch = max(1, floor(size(X_train_spec, 4) / batch_size)); 
    
    % Convert iteration indices to epoch units
    train_epochs = (1:length(info.TrainingLoss)) / iters_per_epoch;
    
    % info.TrainingLoss is recorded every iteration
    plot(train_epochs, info.TrainingLoss, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410], 'DisplayName', 'Training Loss');
    
    % info.ValidationLoss is recorded at validation frequencies
    val_idx = find(~isnan(info.ValidationLoss));
    if ~isempty(val_idx)
        val_epochs = val_idx / iters_per_epoch; % Convert validation iterations to epochs
        plot(val_epochs, info.ValidationLoss(val_idx), '-o', 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Validation Loss');
    end
    
    % --- Set Y-Axis to Logarithmic Scale ---
    set(gca,'XScale', 'log', 'YScale', 'log'); 
    
    xlabel('Epochs', 'FontSize', 14);
    ylabel('MSE Loss (Log Scale)', 'FontSize', 14);
    title(sprintf('CAE Training Curve (k=%d)', bottleNeck), 'FontSize', 16);
    legend('Location', 'northeast', 'FontSize', 12);
    grid on;
    
    % Mark the epoch where early stopping occurred (if applicable)
    if ~isempty(val_idx)
        [min_val_loss, best_idx] = min(info.ValidationLoss(val_idx));
        best_iter = val_idx(best_idx);
        best_epoch = best_iter / iters_per_epoch; % Convert best iteration to epoch
        plot(best_epoch, min_val_loss, 'k*', 'MarkerSize', 10, 'DisplayName', 'Best Model / Early Stop');
    end
    
    saveas(fig_loss, fullfile(method_dir, ['AE_Loss_Curve' file_suffix '.png']));
    close(fig_loss);

    %% Plot 0.5: Evolution of Z-Z_hat Correlation
    if ~isempty(target_epochs)
        fig_corr_evol = figure('Name', 'Correlation Evolution', 'Position', [150, 150, 800, 500], 'Visible', 'off');
        hold on;
        for f = 1:param.N_F
            plot(target_epochs, history_corr(:, f), '-o', 'LineWidth', 1.5, ...
                'Color', h_f_colors(f,:), 'DisplayName', sprintf('Z_{%d}', param.f_peak(f)));
        end
        xlabel('Epochs', 'FontSize', 14);
        ylabel('Pearson Correlation (\rho)', 'FontSize', 14);
        title(['$Z$ vs $\hat{Z}$ Correlation over Training (k=' num2str(bottleNeck) ')'], 'FontSize', 16, 'Interpreter','latex');
        legend('Location', 'eastoutside', 'FontSize', 12);
        grid on;
        ylim([0 1.05]); % Forces Y-axis to standard correlation scale
        
        saveas(fig_corr_evol, fullfile(method_dir, ['AE_Corr_Evolution' file_suffix '.png']));
        close(fig_corr_evol);
    end
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
outAE.history_corr     = history_corr;  % <-- ALWAYS SAVES CORR DATA
outAE.target_epochs    = target_epochs;
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