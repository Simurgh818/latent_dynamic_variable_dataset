function [R2_test, MSE_test, outICA] = runICAAnalysis(eeg_train, eeg_test, h_train, h_test, num_comps, param, method_dir)
% runICAAnalysis EEGLAB ICA + PCA whitening with detailed Frequency Analysis
%
% Inputs:
%   eeg_train, eeg_test : (Channels x Time)
%   h_train, h_test     : (Time x Latents)
%   num_comps           : Number of ICA components (k)
%   param               : Struct with .fs, .N_F, .f_peak
%   param.fs              : Sampling rate for FFT plotting
%   method_dir         : Parent folder for results

%% 1. Setup and Directory

if ~exist(method_dir, 'dir')
    mkdir(method_dir);
end

% File suffix for saving
file_suffix = sprintf('_k%d', num_comps);
h_f_colors = lines(param.N_F); 

%% 2. Run ICA (EEGLAB)
% Prepare EEGLAB Structure
EEG = eeg_emptyset();
EEG.data   = double(eeg_train);
EEG.nbchan = size(eeg_train,1);
EEG.pnts   = size(eeg_train,2);
EEG.trials = 1;
EEG.srate  = param.fs; % Use param.fs if that's the current data rate
EEG.xmin   = 0;
EEG = eeg_checkset(EEG);

% We ensure we don't try to run PCA on more dimensions than exist mathematically.
data_rank = rank(EEG.data); 

% If the requested num_comps is higher than the actual rank, cap it.
% (This prevents ICA from failing if you ever request k > 22)
if num_comps > data_rank
    warning('Requested %d components, but data rank is %d. Caping components.', num_comps, data_rank);
    pca_dim = data_rank;
else
    pca_dim = num_comps;
end

% Run ICA with PCA reduction
% Note: 'pca' flag in pop_runica reduces dimension before ICA
try
    EEG = pop_runica(EEG,'extended', 1, 'pca', pca_dim, 'interrupt','off', 'verbose', 'off'); %
catch
    % Fallback if extended infomax fails or toolbox issues
    warning('Extended ICA failed or not found, trying "runica" default');
    EEG = pop_runica(EEG, 'pca', pca_dim, 'interrupt','off', 'verbose', 'off'); % 
end

% Get Activations
icasig_train = double(EEG.icaact)';   % time x ICs

% Project Test Data
% weights * sphere * data
if ~isempty(EEG.icaweights)
    icasig_test = EEG.icaweights * EEG.icasphere * eeg_test;
    icasig_test = real(icasig_test)';     % time x ICs
else
    % Fallback for some algo implementations
    icasig_test = zeros(size(eeg_test,2), num_comps); 
end

%% 3. Linear Mapping ICs -> Latent Fields
% Reconstruction error per latent on train & test
% R2_test = zeros(EEG.nbchan, param.N_F); 
% MSE_test = zeros(EEG.nbchan, param.N_F); 
% h_rec_test = zeros(size(h_test));
% h_rec_test_norm = zeros(size(h_test));
% 
% for idx = 1:EEG.nbchan
%     for f = 1:param.N_F
%         % fit on train
%        % Reconstruct latent field f from first 'idx' ICs using least-squares
%         x = lsqlin(icasig_train(:,1:idx), h_train(:,f));
%         h_rec_test(:,f) = icasig_test(:,1:idx) * x;
% 
%         % Normalize reconstructed latent variables by their variance (per column)h_rec_test_norm
%         h_rec_test_norm(:,f) = h_rec_test(:,f) ./ std(h_rec_test(:,f), 0, 1);  % Time × N_F
% 
%         numerator = sum((h_test(:,f) - h_rec_test_norm(:,f)).^2);
%         denominator = sum((h_test(:,f) - mean(h_test(:,f))).^2);
%         R2_test(idx, f) = 1 - (numerator / denominator);
%         MSE_test(idx, f) = mean((h_test(:,f) - h_rec_test_norm(:,f)).^2);
%     end
% end

% We learn a linear map A to get from ICs to True Latents
A = pinv(icasig_train) * h_train;

h_rec_train = icasig_train * A;
h_rec_test  = icasig_test  * A;

%% 4. Metrics
% Compute Global R2/MSE on Test Set
MSE_test_vec = mean((h_test - h_rec_test).^2, 1);
MSE_test = mean(MSE_test_vec);

SS_res = sum((h_test(:) - h_rec_test(:)).^2);
SS_tot = sum((h_test(:) - mean(h_test(:))).^2);
R2_test = 1 - (SS_res / SS_tot);

% Compute Zero-Lag Correlation (on Train data for the Trace plot)
zeroLagCorr_ica = zeros(1, param.N_F);
for f = 1:param.N_F
    c = corrcoef(h_test(:,f), h_rec_test(:,f));
    zeroLagCorr_ica(f) = c(1,2);
end

%% 5. Frequency Analysis (FFT Calculation)
% We calculate this on Training data to analyze signal capture fidelity
N = size(h_test, 1);
trial_dur = 1; 
L = round(trial_dur * param.fs);
nTrials = floor(N/L);
f_freq = (0:L-1)*(param.fs/L);
nHz = L/2 + 1;
f_plot = f_freq(1:nHz);

Ht = zeros(L, param.N_F, nTrials);
Hr = zeros(L, param.N_F, nTrials);
R2_trials = zeros(L, param.N_F, nTrials);

for tr = 1:nTrials
    idx = (tr-1)*L + (1:L);
    Z_true_sub = h_test(idx, :);
    Z_recon_sub = h_rec_test(idx, :);
    
    Ht(:,:,tr) = fft(Z_true_sub);
    Hr(:,:,tr) = fft(Z_recon_sub);
    
    for fidx = 1:param.N_F
        num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
        den = abs(Ht(:,fidx,tr)).^2 + eps;
        R2_trials(:,fidx,tr) = 1 - num./den;
    end
end

Ht_avg_ica = mean(Ht, 3);
Hr_avg_ica = mean(Hr, 3);
R2_avg_ica = mean(R2_trials, 3);

% Band Averaging
bands = struct('delta', [1 4], 'theta', [4 8], 'alpha', [8 13], 'beta', [13 30], 'gamma', [30 50]);
band_names = fieldnames(bands);
nBands = numel(band_names);
band_avg_R2_ica = zeros(nBands, param.N_F);

for b = 1:nBands
    f_range = bands.(band_names{b});
    idx_b = f_freq >= f_range(1) & f_freq <= f_range(2);
    for fidx = 1:param.N_F
        band_avg_R2_ica(b, fidx) = mean(R2_avg_ica(idx_b, fidx));
    end
end

% ============================================================
% PLOTTING SECTION
% ============================================================

%% Plot 1: Time Domain + Zero Lag Corr
fig1 = figure('Position',[50 50 1200 150*param.N_F]);
tiledlayout(param.N_F, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['ICA (k=' num2str(num_comps) ') Latent variables Z(t) and $\hat{z}(t)$'], 'Interpreter', 'latex')

for f=1:param.N_F
    nexttile
    hold on;
    set(gca, 'XColor', 'none', 'YColor', 'none'); box on
    % Plot True
    plot(h_test(:, f),'LineStyle', '-', 'Color', h_f_colors(f, :),'DisplayName', ['$Z_{' num2str(param.f_peak(f)) '}$ (t) ']);
    % Plot Recon
    plot(h_rec_test(:, f), 'LineStyle', '--','Color', 'k','DisplayName', ['$\hat{Z}_{' num2str(param.f_peak(f)) '}$ (t) ']);
    
    ylabel('Amp');
    xlim([0 param.fs*2]); % 2 seconds snapshot
    legend('Show','Interpreter', 'latex', 'Location','eastoutside');
    
    rho = zeroLagCorr_ica(f);
    text(0.02 * param.fs, 0.7 * max(abs(h_train(:,f))), ...
        sprintf('\\rho(0)=%.2f', rho), ...
        'FontSize', 12, 'FontWeight', 'bold',...
        'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
        'Margin', 3, 'EdgeColor','k');
    hold off;
end

% Scale bars on last tile
x0 = 0; y0 = min(ylim)+0.2;
line([x0 x0+(param.fs)], [y0 y0], 'Color', 'k', 'LineWidth', 2,'HandleVisibility', 'off');
text(x0+param.fs, y0-0.1, '1 sec', 'VerticalAlignment','top');
line([x0 x0], [y0 y0+2], 'Color', 'k', 'LineWidth', 2,'HandleVisibility', 'off');
text(x0-5, y0+1, '2 a.u.', 'VerticalAlignment','bottom','HorizontalAlignment','right');
set(findall(gcf,'-property','FontSize'),'FontSize',16);
saveas(fig1, fullfile(method_dir, ['ICA_TimeDomain' file_suffix '.png']));


%% Plot 2: Band-wise R2 Bar Chart
fig2 = figure('Position',[50 50 1000 300]);
tiledlayout(1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
bar(band_avg_R2_ica');
set(gca, 'XTickLabel', arrayfun(@(i) sprintf('Z_{%s}', num2str(param.f_peak(i))), 1:param.N_F, 'UniformOutput', false));
ylim([-1 1]);
legend(band_names, 'Location', 'southeastoutside');
ylabel('Mean R^2 in Band'); xlabel('Latent Variable');
title('ICA Band-wise Average R^2(f)');
grid on;
set(findall(gcf,'-property','FontSize'),'FontSize',16);
saveas(fig2, fullfile(method_dir, ['ICA_Bandwise_R2' file_suffix '.png']));


%% Plot 3: Scatter plot: True vs Reconstructed Band Amplitudes (Mean)
Ht_amp_ica = abs(Ht_avg_ica(1:nHz, :));  
Hr_amp_ica = abs(Hr_avg_ica(1:nHz, :)); 

% Normalizing amplitudes
Ht_amp_ica = Ht_amp_ica ./ max(Ht_amp_ica(:));
Hr_amp_ica = Hr_amp_ica ./ max(Hr_amp_ica(:));

mean_band_amp_ica_true  = zeros(nBands, param.N_F);
mean_band_amp_ica_recon = zeros(nBands, param.N_F);
stdDev_band_amp_ica_true  = zeros(nBands, param.N_F);
stdDev_band_amp_ica_recon = zeros(nBands, param.N_F);

for b = 1:nBands
    band = band_names{b};
    f_range = bands.(band);
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);

    mean_band_amp_ica_true(b,:)  = mean(Ht_amp_ica(idx_band,:), 1, 'omitnan');
    mean_band_amp_ica_recon(b,:) = mean(Hr_amp_ica(idx_band,:), 1, 'omitnan');
    stdDev_band_amp_ica_true(b,:)  = std(Ht_amp_ica(idx_band,:), 0, 1, 'omitnan');
    stdDev_band_amp_ica_recon(b,:) = std(Hr_amp_ica(idx_band,:), 0, 1, 'omitnan');
end

true_vals_ica  = mean_band_amp_ica_true(:);
recon_vals_ica = mean_band_amp_ica_recon(:);
band_labels = repelem(band_names, param.N_F);

fig3 = figure('Position',[50 50 1400 300]);
tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('ICA True vs Reconstructed Band Mean FFT Amplitudes');

colors = lines(nBands);
markers = {'o','s','d','h','^','hexagram','<','>'};

for b = 1:nBands    
    nexttile; hold on; 
    idx_b = strcmp(band_labels, band_names{b});
    x_ica = true_vals_ica(idx_b);
    y_ica = recon_vals_ica(idx_b);
    
    for m = 1:length(markers)
        if m > numel(x_ica), break; end
        scatter(x_ica(m), y_ica(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
        errorbar(x_ica(m), y_ica(m), stdDev_band_amp_ica_true(b, m), stdDev_band_amp_ica_recon(b, m), ...
                 'LineStyle', 'none', 'Color', colors(b,:), 'CapSize', 5,'HandleVisibility', 'off');
    end

    xfit_ica = linspace(min(x_ica), max(x_ica), 100);
    plot(xfit_ica, xfit_ica, 'Color', colors(b,:), 'LineWidth', 2, 'DisplayName', "y=x");

    R_fit_ica = corrcoef(x_ica, y_ica);
    R2_fit_ica = R_fit_ica(1,2)^2;
    text(mean(x_ica), mean(y_ica), sprintf('R^2=%.2f', R2_fit_ica), 'Color', colors(b,:), 'FontSize', 12)
    if b==1, xlabel('Mean True Amp.'); ylabel('Mean Recon. Amp.'); end
    title([band_names{b} ' band']); grid on;
end

% Proxy Legend
nLatents = length(markers);
proxy_handles = gobjects(nLatents + 1,1);
for m = 1:nLatents
    proxy_handles(m) = scatter(nan, nan, 70, 'Marker', markers{m}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
end
proxy_handles(end) = plot(nan, nan, 'k', 'LineWidth', 2);
legend_labels = [arrayfun(@(m) sprintf('Z_{%s}', num2str(param.f_peak(m))), 1:length(markers), 'UniformOutput', false), {'y = x'}];
legend(proxy_handles, legend_labels, 'Location','eastoutside','TextColor','k','IconColumnWidth',7, 'NumColumns',2);
hold off;
set(findall(gcf,'-property','FontSize'),'FontSize',14)
saveas(fig3, fullfile(method_dir, ['ICA_Scatter_Mean' file_suffix '.png']));


%% Plot 4: Scatter plot: True vs Reconstructed Band Amplitudes (per trial)
Ht_amp_trials = abs(Ht(1:nHz, :, :)); 
Hr_amp_trials = abs(Hr(1:nHz, :, :));

Ht_amp_trials = Ht_amp_trials ./ max(Ht_amp_trials(:));
Hr_amp_trials = Hr_amp_trials ./ max(Hr_amp_trials(:));

true_vals_band  = cell(nBands, 1);
recon_vals_band = cell(nBands, 1);

for b = 1:nBands
    band = band_names{b};
    f_range = bands.(band);
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);

    temp_true  = squeeze(mean(Ht_amp_trials(idx_band, :, :), 1, 'omitnan'));
    temp_recon = squeeze(mean(Hr_amp_trials(idx_band, :, :), 1, 'omitnan'));

    true_vals_band{b}  = temp_true(:);
    recon_vals_band{b} = temp_recon(:);
end

fig4 = figure('Position',[50 50 1200 300]);
tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('ICA True vs Reconstructed FFT Band Amplitudes (All Trials)');

for b = 1:nBands
    nexttile; hold on;
    x = true_vals_band{b};
    y = recon_vals_band{b};

    scatter(x, y, 30, 'Marker', markers{b}, 'MarkerEdgeColor', colors(b,:), ...
        'MarkerFaceColor', colors(b,:), 'MarkerFaceAlpha', 0.3, ...
        'DisplayName', [sprintf('Z_{%d}', param.f_peak(b))]);

    xfit = linspace(min(x), max(x), 100);
    plot(xfit, xfit, 'k--', 'LineWidth', 1.5, 'DisplayName', 'y=x');

    R_fit = corrcoef(x, y);
    if numel(R_fit) > 1
        R2_fit = R_fit(1,2)^2;
        text(mean(x), mean(y), sprintf('R^2=%.2f', R2_fit), 'Color', 'k', 'FontSize', 12);
    end

    title([band_names{b} ' band']);
    if b==1, xlabel('True Band Amp.'); ylabel('Recon Band Amp.'); end
    grid on; hold off;
end
legend('Location','southoutside','TextColor','k','Orientation','horizontal');
set(findall(gcf,'-property','FontSize'),'FontSize',14)
saveas(fig4, fullfile(method_dir, ['ICA_Scatter_Trials' file_suffix '.png']));


%% 6. Outputs
outICA = struct();
outICA.icasig_train = icasig_train;
outICA.icasig_test  = icasig_test;
outICA.A            = A;
outICA.h_rec_train  = h_rec_train;
outICA.h_rec_test   = h_rec_test;
outICA.MSE_train    = mean((h_test - h_rec_test).^2, 'all');
outICA.method_dir  = method_dir;

close all;
end