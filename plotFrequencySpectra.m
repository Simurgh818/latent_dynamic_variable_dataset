function [Ht, Hr, R2_avg, f_axis, f_plot] = plotFrequencySpectra(Z_true, Z_recon, method_name, param, k, save_path)
    % Z_true: T x N_F (Original)
    % Z_recon: T x N_F (Reconstructed)
    % param: struct with fields .fs, .f_peak, .N_F
    
    T = size(Z_true, 1);
    num_f = size(Z_true, 2);
    trial_dur = 1; % 1 second trials
    L = round(trial_dur * param.fs);
    nTrials = floor(T/L);
    
    f_axis = (0:L-1)*(param.fs/L);
    nHz = floor(L/2) + 1;
    f_plot = f_axis(1:nHz);
    
    Ht = zeros(L, num_f, nTrials);
    Hr = zeros(L, num_f, nTrials);
    R2_trials = zeros(L, param.N_F, nTrials);
    
    % --- FFT Calculation ---
    for tr = 1:nTrials
        idx = (tr-1)*L + (1:L);
        Ht(:,:,tr) = fft(Z_true(idx, :));
        Hr(:,:,tr) = fft(Z_recon(idx, :));
         for fidx = 1:param.N_F
            num = abs(Ht(:,fidx,tr) - Hr(:,fidx,tr)).^2;
            den = abs(Ht(:,fidx,tr)).^2 + eps;
            R2_trials(:,fidx,tr) = 1 - num./den;
         end
    end
    
    % Average spectra (magnitude) across trials
    Ht_avg = mean(abs(Ht(1:nHz, :, :)), 3);
    Hr_avg = mean(abs(Hr(1:nHz, :, :)), 3);
    R2_avg = mean(R2_trials, 3);
    
    % --- Plotting ---
    fig = figure('Position', [100, 100, 1000, 600], 'Visible', 'off');
    tiledlayout(2, 1, 'TileSpacing', 'compact' , 'Padding', 'compact'); %
    colors = lines(num_f);
    
    % Plot 1: True Spectrum
    ax1 = nexttile; 
    for i = 1:num_f
        loglog(f_plot, Ht_avg(:,i), 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('$Z_{%s}(f)$', num2str(param.f_peak(i))));
        hold on;
    end
    title('FFT Amplitude Original'); 
    ylabel('$|Z(f)|$', 'Interpreter', 'latex'); 
    grid on; box on;
    legend('show', 'Location', 'eastoutside', 'Interpreter', 'latex');
    
    % Plot 2: Recon Spectrum
    ax2 = nexttile; 
    for i = 1:num_f
        loglog(f_plot, Hr_avg(:,i), 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('$\\hat{Z}_{%s}(f)$', num2str(param.f_peak(i))));
        hold on;
    end
    title(sprintf('%s FFT Amplitude Reconstructed, k= %d', method_name, k)); 
    xlabel('Frequency (Hz)'); 
    ylabel('$\hat{Z}(f)$', 'Interpreter', 'latex');
    xticks(param.f_peak);
    grid on; box on;
    legend('show', 'Location', 'eastoutside', 'Interpreter', 'latex');

    % --- Crucial Log-Scale Formatting ---
    % 1. Link axes so they share the same scale
    linkaxes([ax1, ax2], 'xy');
    
    % 2. Set Frequency Limits (1Hz to ~100Hz or Nyquist)
    xticks(param.f_peak);
    xlim([1, 100]);
    
    % 3. Set Amplitude Limits (Avoid 0 for log scales)
    % Automatically determine a reasonable bottom (e.g., 2 orders of magnitude below max)
    max_val = max(Ht_avg(:));
    ylim([max_val/1000, max_val]); 
    
    set(findall(fig,'-property','FontSize'),'FontSize',14);
    
    saveas(fig, save_path);
    close(fig);
end