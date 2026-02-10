function [R2_dpca_avg, MSE_dpca_avg, outDPCA] = rundPCAAnalysis( ...
        s_eeg_ds, h_f_normalized_ds, param, k, results_dir)
% rundPCAAnalysis: runs dPCA for a specific k, reconstructs latents, 
% and computes performance metrics.
%
% Inputs:
%   s_eeg_ds             : nChannels x T
%   h_f_normalized_ds    : T x N_F
%   param                : struct (uses param.f_peak)
%   k                    : scalar (Number of components to use)
%   results_dir          : directory to save figures
%
% Outputs:
%   R2_dpca_avg          : Scalar average R^2 across all latents
%   MSE_dpca_avg         : Scalar average MSE across all latents
%   outDPCA              : Structure with detailed results

%% 1. Setup
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
file_suffix = sprintf('_k%d', k);
num_f = size(h_f_normalized_ds, 2);

%% 2. Run dPCA (Single Condition)
% Prepare X for dPCA: (Channels x Time x Trials) -> Trials=1
X_dpca = zeros(size(s_eeg_ds,1), size(s_eeg_ds,2), 1);
X_dpca(:,:,1) = s_eeg_ds;

% Run dPCA for exactly 'k' components
% Note: If your dPCA library supports requesting specific k, this is fine.
% If it returns more, we slice it later.
[W, V, whichMarg] = dpca(X_dpca, k); 

% Latent time series (Components x Time)
Z_dpca = W' * s_eeg_ds;         
Z_dpca_T = Z_dpca';             % T x nComp

% Match components to latents
H = h_f_normalized_ds(1:size(Z_dpca_T,1), :);
[corr_dPCA, R_dPCA] = match_components_to_latents(Z_dpca_T, H, 'dPCA', k);

%% 3. Reconstruction & Metrics (No inner loop!)
% We use ALL k components to reconstruct the latents
h_f_recon_dpca = zeros(size(h_f_normalized_ds));
h_f_recon_normalized_dpca = zeros(size(h_f_normalized_ds));

R2_feat  = zeros(1, num_f);
MSE_feat = zeros(1, num_f);

for f = 1:num_f
    % Linear regression: Z_dpca_T * w = h_f
    % We use all columns (1:k) of Z_dpca_T
    w = Z_dpca_T \ h_f_normalized_ds(:,f);
    
    % Reconstruction
    rec_raw = Z_dpca_T * w;
    h_f_recon_dpca(:,f) = rec_raw;
    
    % --- NORMALIZATION STEP ---
    % Normalize so std=1, matching the ground truth
    rec_norm = rec_raw ./ std(rec_raw);
    h_f_recon_normalized_dpca(:,f) = rec_norm;
    
    % Compute Metrics
    res_var = sum((h_f_normalized_ds(:,f) - rec_norm).^2);
    tot_var = sum((h_f_normalized_ds(:,f) - mean(h_f_normalized_ds(:,f))).^2);
    
    R2_feat(f)  = 1 - (res_var / tot_var);
    MSE_feat(f) = mean((h_f_normalized_ds(:,f) - rec_norm).^2);
end

% Averages to return to main script
R2_dpca_avg  = mean(R2_feat);
MSE_dpca_avg = mean(MSE_feat);

%% 4. Zero-lag correlation (for plotting)
maxLag = 200;
lags   = -maxLag:maxLag;
zeroLagCorr_dpca = zeros(1, num_f);
for f = 1:num_f
    c = xcorr(h_f_normalized_ds(:,f), h_f_recon_normalized_dpca(:,f), maxLag, 'coeff');
    zeroLagCorr_dpca(f) = c(lags==0);
end

%% 5. Explained Variance
[explainedVar_frac, explainedVar_pct, explainedVar_cum] = ...
    dpca_explained_variance(X_dpca, W, V);

%% ============================================================
%  PLOTTING SECTION
% ============================================================
% Only plot if running serially (main thread) and k is large enough
if isempty(getCurrentTask()) && k > 4
    
    % 1. Time Domain Reconstruction
    plotTimeDomainReconstruction(h_f_normalized_ds, h_f_recon_normalized_dpca, ...
        param, 'dPCA', k, zeroLagCorr_dpca, results_dir);
    
    % 2. Component Traces
    plotCTraces(k, param, Z_dpca', results_dir, file_suffix);
    
    % 3. Explained Variance
    % fig3 = figure('Position',[50 50 800 600]);
    % tiledlayout(2, 1, 'Padding', 'compact');
    % nexttile;
    % bar(explainedVar_pct, 'FaceColor', [0.3 0.3 0.9]);
    % ylabel('Explained Variance (%)'); xlabel('Component index');
    % title(['dPCA Variance (k=' num2str(k) ')']);
    % 
    % nexttile;
    % plot(explainedVar_cum, 'LineWidth', 2);
    % ylabel('Cumulative Variance (%)'); xlabel('Component index');
    % ylim([0 100]); title('Cumulative Explained Variance'); grid on;
    % saveas(fig3, fullfile(results_dir,['dPCA_ExplainedVariance' file_suffix '.png'])); 
    % close(fig3);
    save_path = fullfile(results_dir, ['dPCA_ExplainedVariance' file_suffix '.png']);
    plotCumulativeVariance(explainedVar_pct, k, 'dPCA', save_path);

    % --- Frequency Analysis Prep ---
    Z_true = h_f_normalized_ds;
    Z_rec  = h_f_recon_normalized_dpca;
    
    N = size(Z_true,1);
    trial_dur = 1;         
    L = round(trial_dur * param.fs);
    nTrials = floor(N/L);
    f_axis = (0:L-1)*(param.fs/L);
    nHz = L/2 + 1;
    f_plot = f_axis(1:nHz);
    
    % FFT Calculation
    Ht = zeros(L, num_f, nTrials);
    Hr = zeros(L, num_f, nTrials);
    R2_trials = zeros(L, num_f, nTrials);
    
    for tr = 1:nTrials
        idx = (tr-1)*L + (1:L);
        Ht(:,:,tr) = fft(Z_true(idx,:));
        Hr(:,:,tr) = fft(Z_rec(idx,:));
        for fidx = 1:num_f
            num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
            den = abs(Ht(:,fidx,tr)).^2 + eps;
            R2_trials(:,fidx,tr) = 1 - num./den;
        end
    end
    
    Ht_avg = mean(Ht,3);
    Hr_avg = mean(Hr,3);
    R2_avg_freq = mean(R2_trials,3);
    
    % 4. FFT Comparison Plot
    h_f_colors = lines(num_f);
    fig4 = figure('Position',[50 50 1200 600]);
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
    
    nexttile;
    for fidx=1:num_f
        loglog(f_plot, abs(Ht_avg(1:nHz,fidx)), 'Color', h_f_colors(fidx,:), ...
            'DisplayName', sprintf("Z_{%s}(f)", num2str(param.f_peak(fidx))));
        hold on;
    end
    title('FFT of Original Latents'); legend('show','Location','eastoutside'); grid on;
    
    nexttile;
    for fidx=1:num_f
        loglog(f_plot, abs(Hr_avg(1:nHz,fidx)), 'Color', h_f_colors(fidx,:), ...
            'DisplayName', sprintf("\\hat{Z}_{%s}(f)", num2str(param.f_peak(fidx))));
        hold on;
    end
    title('FFT of Reconstructed Latents'); legend('show','Location','eastoutside'); grid on;
    saveas(fig4, fullfile(results_dir,['dPCA_FFT_True_vs_Recon' file_suffix '.png']));
    close(fig4);
    
    % 5. Band-wise R2 Bar Chart
    bands = struct('delta',[1 4],'theta',[4 8],'alpha',[8 13], ...
                   'beta',[13 30],'gamma',[30 50]);
    band_names = fieldnames(bands);
    nBands = numel(band_names);
    band_avg_R2 = zeros(nBands, num_f);
    
    for b = 1:nBands
        f_range = bands.(band_names{b});
        idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
        for fidx=1:num_f
            band_avg_R2(b,fidx) = mean(R2_avg_freq(idx_band,fidx));
        end
    end
    
    fig5 = figure('Position',[50 50 1000 300]);
    bar(band_avg_R2');
    set(gca,'XTickLabel', arrayfun(@(i) sprintf('Z_{%s}', num2str(param.f_peak(i))), 1:num_f, 'UniformOutput',false));
    legend(band_names, 'Location','eastoutside'); title('Band-wise R^2 of dPCA Reconstruction');
    ylim([-1 1]); grid on;
    saveas(fig5, fullfile(results_dir,['dPCA_Bandwise_R2' file_suffix '.png'])); 
    close(fig5);
    
    % 6. Scatter Plots
    % (Keeping your scatter logic, simplified call)
    plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, k, "dPCA", results_dir);
end

%% 6. Package Output
outDPCA = struct();
outDPCA.W = W;
outDPCA.V = V;
outDPCA.Z_dpca = Z_dpca;
outDPCA.h_recon_train = h_f_recon_normalized_dpca; % Normalized reconstruction
outDPCA.zeroLagCorr = zeroLagCorr_dpca;
outDPCA.explainedVar_frac = explainedVar_frac;
outDPCA.explainedVar_pct  = explainedVar_pct;
outDPCA.folder = results_dir;
outDPCA.corr_dPCA = corr_dPCA;
outDPCA.R_full = R_dPCA;
outDPCA.R2_features = R2_feat;
outDPCA.MSE_features = MSE_feat;

close all;
end

%% Nested Helper Function
function [explainedVar_frac, explainedVar_pct, explainedVar_cum] = dpca_explained_variance(X, W, V)
    % Ensure shaped as n x (T*C)
    if ismatrix(X)
        Xflat = X;                    
    elseif ndims(X) == 3
        Xflat = reshape(X, size(X,1), []); 
    else
        error('X must be 2D or 3D array');
    end
    % Mean-center 
    Xmean = mean(Xflat, 2);           
    Xc = bsxfun(@minus, Xflat, Xmean);
    % Total variance 
    perChannelVar = var(Xc, 0, 2);    
    totalVar = sum(perChannelVar(:)) + eps;
    % Project to components 
    Z = V' * Xc;                       
    nComp = size(Z,1);
    explainedVar_frac = zeros(1,nComp);
    % Compute contribution 
    for c = 1:nComp
        Recon_c = W(:,c) * Z(c,:);    
        explainedVar_frac(c) = sum(var(Recon_c, 0, 2)) / totalVar;
    end
    % Normalize and convert to percent
    explainedVar_frac = explainedVar_frac ./ sum(explainedVar_frac); 
    explainedVar_pct  = 100 * explainedVar_frac;
    explainedVar_cum  = cumsum(explainedVar_pct);
end