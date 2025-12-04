function [R2_dpca, MSE_dpca, outDPCA] = rundPCAAnalysis( ...
        s_eeg_ds, h_f_normalized_ds, param, num_sig_components, results_dir)
% rundPCAAnalysis: runs single-condition dPCA, reconstructs latents, computes
% R^2/MSE, makes diagnostic plots and saves them to results_dir.
%
% Inputs:
%   s_eeg_ds             : nChannels × T
%   h_f_normalized_ds    : T × N_F
%   param                : struct (uses param.f_peak)
%   num_sig_components   : scalar
%   results_dir          : directory to save figures
%   param.fs               : sampling rate used for plotting scale bars
%
% Outputs:
%   R2_dpca, MSE_dpca, outDPCA


% --- simple arg checks / ensure folder exists
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% File naming suffix
file_suffix = sprintf('k%d', num_sig_components);

% 1) Prepare X for dPCA (single condition)
X_dpca = zeros(size(s_eeg_ds,1), size(s_eeg_ds,2), 1);
X_dpca(:,:,1) = s_eeg_ds;

% 2) Run dPCA
[W, V, whichMarg] = dpca(X_dpca, num_sig_components);

% 2b) latent time series (component × time)
Z_dpca = W' * s_eeg_ds;         % nComp × T
Z_dpca_T = Z_dpca';             % T × nComp

% 3) Reconstruct original h_f using lsqlin
T = size(s_eeg_ds,2);
num_f = size(h_f_normalized_ds,2);

h_f_recon_dpca = zeros(T, num_f);
h_f_recon_normalized_dpca = zeros(T, num_f);

R2_dpca  = zeros(num_sig_components, num_f);
MSE_dpca = zeros(num_sig_components, num_f);

for idx = 1:num_sig_components
    for f = 1:num_f
        % lsqlin fit: Z_dpca_T(:,1:idx) -> h_f
        w = lsqlin(Z_dpca_T(:,1:idx), h_f_normalized_ds(:,f));
        % reconstruction
        h_f_recon_dpca(:,f) = Z_dpca_T(:,1:idx) * w;
        % normalize reconstructed latent
        h_f_recon_normalized_dpca(:,f) = h_f_recon_dpca(:,f) ./ std(h_f_recon_dpca(:,f));
        % compute R^2
        numerator   = sum((h_f_normalized_ds(:,f) - h_f_recon_normalized_dpca(:,f)).^2);
        denominator = sum((h_f_normalized_ds(:,f) - mean(h_f_normalized_ds(:,f))).^2);
        R2_dpca(idx,f) = 1 - numerator/denominator;
        % compute MSE
        MSE_dpca(idx,f) = mean((h_f_normalized_ds(:,f) - h_f_recon_normalized_dpca(:,f)).^2);
    end
end

% 4) Zero-lag correlation
maxLag = 200;
lags   = -maxLag:maxLag;
zeroLagCorr_dpca = zeros(1, num_f);
for f = 1:num_f
    c = xcorr(h_f_normalized_ds(:,f), h_f_recon_dpca(:,f), maxLag, 'coeff');
    zeroLagCorr_dpca(f) = c(lags==0);
end

% 5) Explained variance
[explainedVar_frac, explainedVar_pct, explainedVar_cum] = ...
    dpca_explained_variance(X_dpca, W, V);

% colors for plotting
h_f_colors = lines(num_f);

%%% -------------------- Plot reconstructed latent signals --------------------
fig1 = figure('Position',[50 50 1200 150*num_f]);
tiledlayout(num_f, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['Latent variables Z(t) and dPCA $\hat{z}$(t) reconstruction. (k=' ...
    num2str(num_sig_components) ')'], 'Interpreter','latex')

for f=1:num_f
    nexttile;
    hold on;
    set(gca, 'XColor', 'none', 'YColor', 'none');
    box on
    plot(h_f_normalized_ds(:, f),'LineStyle', '-', 'Color', h_f_colors(f, :), ...
        'DisplayName', ['Z_{' num2str(param.f_peak(f)) '} (true)']);
    plot(h_f_recon_dpca(:, f), 'LineStyle', '--','LineWidth',1,'Color', 'k', ...
        'DisplayName', ['Z_{' num2str(param.f_peak(f)) '} (recon)']);
    ylabel('amplitude')
    xlim([0 param.fs*2]);
    legend('Show','Location','eastoutside');
    rho = zeroLagCorr_dpca(f);
    text(0.02 * param.fs, 0.05 * max(h_f_normalized_ds(:,f)), ...
        sprintf('\\rho(0)=%.2f', rho), ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
        'Margin', 3, 'EdgeColor','k');
    hold off;
end
% scale bars (draw on last axis)
ax = gca;
hold(ax,'on');
x0 = 0;
y0 = min(ylim)+0.2;
line([x0 x0+param.fs], [y0 y0], 'Color', 'k', 'LineWidth', 2);
text(x0+param.fs, y0-0.1, '1 sec', 'VerticalAlignment','top');
line([x0 x0], [y0 y0+2], 'Color', 'k', 'LineWidth', 2);
text(x0-5, y0+4, '2 a.u.', 'VerticalAlignment','bottom', ...
    'HorizontalAlignment','right','Rotation',90);
set(findall(fig1,'-property','FontSize'),'FontSize',16);
saveas(fig1, fullfile(results_dir,['dPCA_TimeDomain_Reconstruction_k' file_suffix '.png'])); 

%%% ---------------------- Plot Z_dpca component traces ------------------------
fig2 = figure('Position',[50 50 1000 (num_sig_components*250)/2]);
tiledlayout(num_sig_components, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
pc_colors = lines(num_sig_components);
sgtitle(['PC Traces (k=' num2str(num_sig_components) ')'])

for pc = 1:num_sig_components
    nexttile;
    plot(Z_dpca(pc,:), 'LineStyle', '-', 'Color', pc_colors(pc,:), ...
        'DisplayName', ['PC(t) ' num2str(pc)]);
    xlabel('Time bins')
    ylabel('PC amplitude')
    xlim([0 1000]);
    legend('show');
end
set(findall(fig2,'-property','FontSize'),'FontSize',12);
saveas(fig2, fullfile(results_dir,['dPCA_Component_Traces_k' file_suffix '.png'])); 

%%% -------------------- Explained variance figure -----------------------------
fig3 = figure('Position',[50 50 800 600]);
tiledlayout(2, 1, 'Padding', 'compact');
% Fraction per component
nexttile
bar(explainedVar_pct, 'FaceColor', [0.3 0.3 0.9])
ylabel('Explained Variance (%)')
xlabel('Component index')
title(['dPCA Component Variance Explained(k=' num2str(num_sig_components) ')'])
% Cumulative variance
nexttile
plot(explainedVar_cum, 'LineWidth', 2)
ylabel('Cumulative Variance (%)')
xlabel('Component index')
ylim([0 100])
title('Cumulative Explained Variance')
grid on
set(findall(fig3,'-property','FontSize'),'FontSize',12);
saveas(fig3, fullfile(results_dir,['dPCA_ExplainedVariance_k' file_suffix '.png'])); 

%% --- Frequency-domain analysis for dPCA reconstruction ---
Z_true = h_f_normalized_ds;
Z_rec  = h_f_recon_dpca;

N = size(Z_true,1);
trial_dur = 1;         % seconds per synthetic "trial"
L = round(trial_dur * param.fs);
nTrials = floor(N/L);
f = (0:L-1)*(param.fs/L);
nF = size(Z_true,2);

% Storage
R2_trials = zeros(L, nF, nTrials);
Ht = zeros(L, nF, nTrials);
Hr = zeros(L, nF, nTrials);

Zt_trials = zeros(L, nTrials, nF);
Zr_trials = zeros(L, nTrials, nF);

for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);
    Zt_trials(:,tr,:) = Z_true(idx,:);
    Zr_trials(:,tr,:) = Z_rec(idx,:);
    Ht(:,:,tr) = fft(Zt_trials(:,tr,:));
    Hr(:,:,tr) = fft(Zr_trials(:,tr,:));
    for fidx = 1:nF
        num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
        den = abs(Ht(:,fidx,tr)).^2 + eps;
        R2_trials(:,fidx,tr) = 1 - num./den;
    end
end

% Averages
R2_avg = mean(R2_trials,3);
Ht_avg = mean(Ht,3);
Hr_avg = mean(Hr,3);

%% --- Scatter plot: True vs Reconstructed Band Amplitudes ---
% Compute mean FFT amplitude in each band for true vs reconstructed latents
nHz = L/2+1;
f_plot = f(1:nHz);

bands = struct( ...
    'delta', [1 4], ...
    'theta', [4 8], ...
    'alpha', [8 13], ...
    'beta',  [13 30], ...
    'gamma', [30 50] ...
);
band_names = fieldnames(bands);
nBands = numel(band_names);

Ht_amp = abs(Ht_avg(1:nHz, :));  % True latent mean amplitude spectrum
Hr_amp = abs(Hr_avg(1:nHz, :));  % Reconstructed latent mean amplitude spectrum

% Normalizing amplitudes before comparison:
Ht_amp = Ht_amp ./ max(Ht_amp(:));
Hr_amp = Hr_amp ./ max(Hr_amp(:));

mean_band_amp_true  = zeros(nBands, size(h_f_normalized_ds,2));
mean_band_amp_recon = zeros(nBands, size(h_f_normalized_ds,2));
stdDev_band_amp_true  = zeros(nBands, size(h_f_normalized_ds,2));
stdDev_band_amp_recon = zeros(nBands, size(h_f_normalized_ds,2));



for b = 1:nBands
    band = band_names{b};
    f_range = bands.(band);
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);

    % mean FFT amplitude within this frequency band
    mean_band_amp_true(b,:)  = mean(Ht_amp(idx_band,:), 1, 'omitnan');
    mean_band_amp_recon(b,:) = mean(Hr_amp(idx_band,:), 1, 'omitnan');

    % Standard Deviation FFT amplitudes
    stdDev_band_amp_true(b,:)  = std(Ht_amp(idx_band,:), 0, 1, 'omitnan');
    stdDev_band_amp_recon(b,:) = std(Hr_amp(idx_band,:), 0, 1, 'omitnan');
end

%%% --- Plot: FFT amplitudes of true vs reconstructed ---
nHz = L/2+1;
f_plot = f(1:nHz);

fig4 = figure('Position',[50 50 1200 600]);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
sgtitle(['dPCA Frequency Analysis (k=' num2str(num_sig_components) ')']);

% True FFT
nexttile;
for fidx=1:nF
    loglog(f_plot, abs(Ht_avg(1:nHz,fidx)), 'Color', h_f_colors(fidx,:), ...
        'DisplayName', sprintf('Z_{%s}(f)', num2str(param.f_peak(fidx))));
    hold on;
end
xlabel('Frequency (Hz)'); ylabel('|Z(f)|');
title('FFT of Original Latents');
xlim([1 50]); xticks([1 4 8 10 13 20 30 50]);
grid on; legend('show','Location','southeastoutside');

% Reconstructed FFT
nexttile;
for fidx=1:nF
    loglog(f_plot, abs(Hr_avg(1:nHz,fidx)), 'Color', h_f_colors(fidx,:), ...
        'DisplayName', sprintf('\\hat{Z}_{%d}(f)', param.f_peak(fidx)));
    hold on;
end
xlabel('Frequency (Hz)'); ylabel('|Ẑ(f)|');
title('FFT of Reconstructed Latents');
xlim([1 50]); xticks([1 4 8 10 13 20 30 50]);
grid on; legend('show','Location','southeastoutside');

set(findall(fig4,'-property','FontSize'),'FontSize',14);
saveas(fig4, fullfile(results_dir,['dPCA_FFT_True_vs_Recon_' file_suffix '.png']));

%%% --- Band-wise R² ---
bands = struct('delta',[1 4],'theta',[4 8],'alpha',[8 13], ...
               'beta',[13 30],'gamma',[30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);
band_avg_R2 = zeros(nBands, nF);
for b = 1:nBands
    f_range = bands.(band_names{b});
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
    for fidx=1:nF
        band_avg_R2(b,fidx) = mean(R2_avg(idx_band,fidx));
    end
end

fig5 = figure('Position',[50 50 1000 300]);
bar(band_avg_R2');
set(gca,'XTickLabel', arrayfun(@(i) sprintf('Z_{%s}', num2str(param.f_peak(i))), 1:nF, 'UniformOutput',false));
ylim([-1 1]);
ylabel('Mean R^2'); xlabel('Latent Variable');
legend(band_names, 'Location','southeastoutside'); title('Band-wise R^2 of dPCA Reconstruction');
grid on; set(findall(fig5,'-property','FontSize'),'FontSize',14);
saveas(fig5, fullfile(results_dir,['dPCA_Bandwise_R2' file_suffix '.png'])); 

%%% --- True vs reconstructed band FFT amplitude scatter ---
% Ht_amp = abs(Ht_avg(1:nHz,:));
% Hr_amp = abs(Hr_avg(1:nHz,:));
% Ht_amp = Ht_amp ./ max(Ht_amp(:));
% Hr_amp = Hr_amp ./ max(Hr_amp(:));
% 
% mean_true = zeros(nBands,nF);
% mean_rec  = zeros(nBands,nF);
% std_true  = zeros(nBands,nF);
% std_rec   = zeros(nBands,nF);
% 
% for b = 1:nBands
%     f_range = bands.(band_names{b});
%     idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);
%     mean_true(b,:) = mean(Ht_amp(idx_band,:),1);
%     mean_rec(b,:)  = mean(Hr_amp(idx_band,:),1);
%     std_true(b,:) = std(Ht_amp(idx_band,:),0,1);
%     std_rec(b,:)  = std(Hr_amp(idx_band,:),0,1);
% end
numF = param.N_F;
% -------------------------------------------------------------
% dPCA Reconstruction: True vs Reconstructed Band Amplitudes
% -------------------------------------------------------------
%% ===================== BAND-WISE TRUE VS RECON PLOTS (dPCA) =====================

true_vals  = mean_band_amp_true(:);
recon_vals = mean_band_amp_recon(:);
band_labels = repelem(band_names, size(h_f_normalized_ds,2));

fig6 = figure('Position',[50 50 1600 300]);
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
            'DisplayName', [sprintf('Z_{%d}', param.f_peak(m))]);
        
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
set(findall(gcf,'-property','FontSize'),'FontSize',14)
saveas(fig6, fullfile(results_dir, ['dPCA_Scatter_Band_Amp_Mean' file_suffix '.png']));

%% ===================== TRIALWISE BAND AMPLITUDE SCATTER PLOTS (dPCA) =====================

% Trialwise FFT amplitudes
Ht_amp_trials = abs(Ht(1:nHz, :, :));              % true latent FFT
Hr_amp_trials_dpca = abs(Hr(1:nHz, :, :));    % dPCA reconstructed FFT

% Normalize across everything exactly like before
Ht_amp_trials = Ht_amp_trials ./ max(Ht_amp_trials(:));
Hr_amp_trials_dpca = Hr_amp_trials_dpca ./ max(Hr_amp_trials_dpca(:));

% Containers for scatter values
true_vals_band  = cell(nBands, 1);
recon_vals_band_dpca = cell(nBands, 1);

for b = 1:nBands
    band = band_names{b};
    f_range = bands.(band);
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);

    % Band-averaged amplitudes per trial × latent
    % size → [nLatent × nTrials]
    temp_true  = squeeze(mean(Ht_amp_trials(idx_band, :, :), 1, 'omitnan'));
    temp_recon = squeeze(mean(Hr_amp_trials_dpca(idx_band, :, :), 1, 'omitnan'));

    % Flatten latent × trial into long vectors
    true_vals_band{b}  = temp_true(:);
    recon_vals_band_dpca{b} = temp_recon(:);
end

%% --------------------- PLOTTING ------------------------
fig7 = figure('Position',[50 50 1400 350]);
tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['True vs dPCA Reconstructed FFT Band Amplitudes (All Trials × Latents), (k=' num2str(num_sig_components) ')']);

colors = lines(nBands);
markers = {'o','s','d','h','^','hexagram','<','>'};

for b = 1:nBands
    nexttile;
    hold on;

    x = true_vals_band{b};
    y = recon_vals_band_dpca{b};

    % Scatter of all trial points
    scatter(x, y, 30, 'Marker', markers{b}, ...
        'MarkerEdgeColor', colors(b,:), ...
        'MarkerFaceColor', colors(b,:), ...
        'MarkerFaceAlpha', 0.3, ...
        'DisplayName', band_names{b});

    % Identity line
    xfit = linspace(min(x), max(x), 100);
    plot(xfit, xfit, 'k--', 'LineWidth', 1.5, 'DisplayName', 'y=x');

    % Regression R²
    Rfit = corrcoef(x, y);
    if numel(Rfit) > 1
        R2fit = Rfit(1,2)^2;
        text(mean(x), mean(y), sprintf('R^2=%.2f', R2fit), ...
            'Color', 'k', 'FontSize', 12);
    end

    title([band_names{b} ' band'])
    if b==1
        xlabel('True Band Amplitude')
        ylabel('Reconstructed Band Amplitude')
    end

    legend('Location','southoutside','TextColor','k','Orientation','horizontal');
    grid on;
    hold off;
end

set(findall(gcf,'-property','FontSize'),'FontSize',14)
saveas(fig7, fullfile(results_dir,['dPCA_BandScatter_perTrial_' file_suffix '.png'])); 
close All;
%% 6) Package output struct
outDPCA.W = W;
outDPCA.V = V;
outDPCA.Z_dpca = Z_dpca;
outDPCA.zeroLagCorr = zeroLagCorr_dpca;
outDPCA.explainedVar_frac = explainedVar_frac;
outDPCA.explainedVar_pct  = explainedVar_pct;
outDPCA.explainedVar_cum  = explainedVar_cum;
outDPCA.folder = results_dir;

% Inputs:
%   X          : nChannels x T  (single condition) OR nChannels x T x C
%   W          : nChannels x nComp  (decoder from dpca)
%   V          : nComp x nChannels  (encoder from dpca) -- sometimes V = W'
% Outputs:
%   explainedVar_frac : 1 x nComp, fraction of total variance explained by each comp
%   explainedVar_pct  : percent
%   explainedVar_cum  : cumulative percent

function [explainedVar_frac, explainedVar_pct, explainedVar_cum] = dpca_explained_variance(X, W, V)

    %--- ensure shaped as n x (T*C) ---
    if ismatrix(X)
        Xflat = X;                    % n x T
    elseif ndims(X) == 3
        Xflat = reshape(X, size(X,1), []);  % n x (T*C)
    else
        error('X must be 2D or 3D array');
    end

    %--- mean-center across time/conditions (important) ---
    Xmean = mean(Xflat, 2);           % n x 1
    Xc = bsxfun(@minus, Xflat, Xmean);

    %--- total variance (sum over channels) ---
    perChannelVar = var(Xc, 0, 2);    % n x 1 (unbiased sample var, normalized by N-1)
    totalVar = sum(perChannelVar(:)) + eps;

    %--- project to components (nComp x Nsamples) ---
    Z = V' * Xc;                       % nComp x (T*C)

    nComp = size(Z,1);
    explainedVar_frac = zeros(1,nComp);

    %--- compute contribution of each component to the original data ---
    for c = 1:nComp
        Recon_c = W(:,c) * Z(c,:);    % n x (T*C)
        % variance explained by this component (sum across channels)
        explainedVar_frac(c) = sum(var(Recon_c, 0, 2)) / totalVar;
    end

    %--- normalize tiny numerical drift, and compute percentages ---
    explainedVar_frac = explainedVar_frac ./ sum(explainedVar_frac); % optional normalize
    explainedVar_pct  = 100 * explainedVar_frac;
    explainedVar_cum  = cumsum(explainedVar_pct);

end

end
