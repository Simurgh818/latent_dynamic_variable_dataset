function [outICA] = runICAAnalysis(eeg_train, eeg_test, h_train, h_test, num_comps, param, method_dir)
% runICAAnalysis EEGLAB ICA + PCA whitening with detailed Frequency Analysis
%
% Inputs:
%   eeg_train, eeg_test : (Channels x Time)
%   h_train, h_test     : (Time x Latents)
%   num_comps           : Number of ICA components (k)
%   param               : Struct with .fs, .N_F, .f_peak
%   method_dir          : Parent folder for results

%% 1. Setup and Directory
if ~exist(method_dir, 'dir')
    mkdir(method_dir);
end
file_suffix = sprintf('_k%d', num_comps);
h_f_colors = lines(param.N_F); 

%% 2. EXPLICIT MEAN CENTERING (Crucial for ICA Stability)
train_mean = mean(eeg_train, 2);
% Using bsxfun ensures matrix subtraction compatibility across all MATLAB versions
eeg_train_centered = bsxfun(@minus, eeg_train, train_mean);
eeg_test_centered  = bsxfun(@minus, eeg_test, train_mean);

%% 3. SAFE RANK CALCULATION
% Compute the rank of the spatial covariance matrix (Channels x Channels)
data_cov  = cov(eeg_train_centered');
data_rank = rank(data_cov); 

if num_comps > data_rank
    warning('Requested %d components, but spatial data rank is %d. Capping components.', num_comps, data_rank);
    k = data_rank;
else
    k = num_comps;
end

%% 4. Run ICA (Switched to EEGLAB's native Infomax ICA)
try
    % runica is the EEGLAB gold standard and handles PCA reduction perfectly
    [weights, sphere] = runica(eeg_train_centered, ... 
        'pca', k, ...
        'maxsteps', 2000);
    
    % The full spatial unmixing matrix (k x Channels)
    W = weights * sphere;
    
    % Project the data into component space and transpose to (Time x k)
    icasig_train = (W * eeg_train_centered)'; 
    icasig_test  = (W * eeg_test_centered)';
    
catch ME
    fprintf('runica failed. Falling back to MATLAB "rica". Error: %s \n', ME.message);
    Mdl = rica(eeg_train_centered', k); 
    icasig_train = transform(Mdl, eeg_train_centered');
    icasig_test  = transform(Mdl, eeg_test_centered');
end

%% 5. Compute Performance Metrics
[h_rec_train, h_rec_test, Comp_latent_matching_corr, R_ICA, direct_Component_Corr_ica, ica_R2_scores, freq_data] = ...
    computePerformanceMetrics(icasig_train, icasig_test, h_train, h_test, 'ICA', num_comps, param);

%% ============================================================
% PLOTTING SECTION (Safely skipped by parallel workers)
% ============================================================
if (isempty(getCurrentTask()) && num_comps==10)
    % Time domain plot
    plotTimeDomainReconstruction(h_test, h_rec_test, param, 'ICA', num_comps, direct_Component_Corr_ica, method_dir);
    % Independent Component traces plot
    plotCTraces(num_comps, param, h_rec_test, method_dir, file_suffix);
    
    % Frequency Analysis FFT
    save_path_fft = fullfile(method_dir, ['ICA_FFT_True_vs_Recon' file_suffix '.png']);
    plotFrequencySpectra(freq_data.Ht_avg, freq_data.Hr_avg, freq_data.f_plot, 'ICA', param, num_comps, save_path_fft);
    
    %% Plot 2: Band-wise R2 Bar Chart
    br2_path = fullfile(method_dir, ['ICA_Bandwise_R2' file_suffix '.png']);
    plotBandwiseR2(freq_data.R2_avg, freq_data.f_axis, param, num_comps, 'ICA', br2_path); 
    
    %% Plot 3: Scatter plot: True vs Reconstructed Band Amplitudes (Mean)
    nBands = numel(freq_data.band_names);
    fig3 = figure('Position',[50 50 1400 300], 'Visible', 'off');
    tiledlayout(1, nBands, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(['ICA True vs Reconstructed Band Mean FFT Amplitudes, (k=' num2str(num_comps) ')']);
    colors = lines(nBands);
    markers = {'o','s','d','h','^','hexagram'};
    
    for b = 1:nBands    
        nexttile; hold on; 
        
        % Extract means and stds directly from freq_data (which holds per-trial data)
        x_ica = mean(freq_data.true_vals{b}, 2, 'omitnan');
        y_ica = mean(freq_data.recon_vals{b}, 2, 'omitnan');
        std_x = std(freq_data.true_vals{b}, 0, 2, 'omitnan');
        std_y = std(freq_data.recon_vals{b}, 0, 2, 'omitnan');
        
        for m = 1:length(markers)
            if m > numel(x_ica), break; end
            scatter(x_ica(m), y_ica(m), 70, 'filled', 'MarkerFaceColor', colors(b,:),'Marker', markers{m});
            errorbar(x_ica(m), y_ica(m), std_x(m), std_y(m), ...
                     'LineStyle', 'none', 'Color', colors(b,:), 'CapSize', 5,'HandleVisibility', 'off');
        end
        
        % Fit line and R2
        plot([min(x_ica) max(x_ica)], [min(x_ica) max(x_ica)], 'Color', colors(b,:), 'LineWidth', 2, 'DisplayName', "y=x");
        R_fit_ica = corrcoef(x_ica, y_ica);
        if numel(R_fit_ica) > 1, R2_fit_ica = R_fit_ica(1,2)^2; else, R2_fit_ica = 0; end
        text(mean(x_ica), mean(y_ica), sprintf('R^2=%.2f', R2_fit_ica), 'Color', colors(b,:), 'FontSize', 12)
        
        if b==1, xlabel('Mean True Amp.'); ylabel('Mean Recon. Amp.'); end
        title([freq_data.band_names{b} ' band']); grid on;
    end
    
    nLatents = length(markers);
    proxy_handles = gobjects(nLatents + 1,1);
    for m = 1:nLatents
        proxy_handles(m) = scatter(nan, nan, 70, 'Marker', markers{m}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
    end
    proxy_handles(end) = plot(nan, nan, 'k', 'LineWidth', 2);
    legend_labels = [arrayfun(@(m) sprintf('Z_{%s}', num2str(param.f_peak(m))), 1:length(markers), 'UniformOutput', false), {'y = x'}];
    legend(proxy_handles, legend_labels, 'Location','eastoutside','TextColor','k','IconColumnWidth',7, 'NumColumns',2);
    hold off;
    
    set(findall(fig3,'-property','FontSize'),'FontSize',14)
    saveas(fig3, fullfile(method_dir, ['ICA_Scatter_Mean' file_suffix '.png']));
    close(fig3);
    
    %% Plot 4: Scatter plot: True vs Reconstructed Band Amplitudes (per trial)  
    plotBandScatterPerTrial(freq_data.true_vals, freq_data.recon_vals, ica_R2_scores, freq_data.band_names, param, num_comps, "ICA", method_dir);
end

%% 6. Outputs
outICA = struct();
outICA.icasig_train = icasig_train;
outICA.icasig_test  = icasig_test;
outICA.h_recon_train = h_rec_train;
outICA.h_recon_test  = h_rec_test;
outICA.matched_R2       = freq_data.matched_R2;
outICA.method_dir   = method_dir;
outICA.Comp_latent_matching_corr = Comp_latent_matching_corr;
outICA.direct_Component_Corr = direct_Component_Corr_ica;
outICA.Comp_latent_matching_matrix = R_ICA; 
outICA.spectral_R2  = ica_R2_scores;
close all;

end