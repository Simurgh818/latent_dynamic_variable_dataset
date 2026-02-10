function [Ht, Hr, R2_avg, f_axis, f_plot] = plotFrequencySpectra(Z_true, Z_recon, method_name, param, save_path)
    % Z_true: T x N_F (Original)
    % Z_recon: T x N_F (Reconstructed)
    
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
    
    Ht_avg = mean(abs(Ht(1:nHz, :, :)), 3);
    Hr_avg = mean(abs(Hr(1:nHz, :, :)), 3);
    R2_avg = mean(R2_trials, 3);

    % Plotting
    fig = figure('Position', [100, 100, 1000, 600], 'Visible', 'off');
    tiledlayout(2, 1, 'TileSpacing', 'compact');
    colors = lines(num_f);
    
    % True Spectrum
    nexttile; hold on;
    for i = 1:num_f
        loglog(f_plot, abs(Ht_avg(1:nHz,i)), 'Color', colors(i,:), ...
            'DisplayName', sprintf("Z_{%s}(f)", num2str(param.f_peak(i))));
    end
    xlim([1 101]);
    ylim([0 max(Ht_avg(1:nHz,1))]);
    title('Ground Truth Latent Spectra'); ylabel('Magnitude'); 
    legend('show','Location','eastoutside', 'Interpreter','latex'); grid on;
    
    % Recon Spectrum
    nexttile; hold on;
    for i = 1:num_f
        loglog(f_plot, abs(Hr_avg(1:nHz,i)), 'Color', colors(i,:), ...
            'DisplayName', sprintf("\\hat{Z}_{%s}(f)", num2str(param.f_peak(i))));
    end
    xlim([1 101]);
    ylim([0 max(Ht_avg(1:nHz,1))]);
    title([method_name ' Reconstructed Spectra']); xlabel('Frequency (Hz)'); ylabel('Magnitude'); 
    legend('show','Location','eastoutside', 'Interpreter','latex'); grid on;
    
    saveas(fig, save_path);
    close(fig);
end