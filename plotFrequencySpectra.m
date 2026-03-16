function plotFrequencySpectra(Ht_avg, Hr_avg, f_plot, method_name, param, k, save_path)
    num_f = size(Ht_avg, 2);
    
    fig = figure('Position', [100, 100, 1000, 800], 'Visible', 'off');
    tiledlayout(2, 1, 'TileSpacing', 'compact' , 'Padding', 'compact'); 
    colors = lines(num_f);
    
    % Plot 1: True Spectrum
    ax1 = nexttile; 
    for i = 1:num_f
        loglog(f_plot, Ht_avg(:,i), 'Color', colors(i,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('$Z_{%s}(f)$', num2str(param.f_peak(i))));
        hold on;
    end
    title('FFT Amplitude Original'); 
    xticks(param.f_peak);
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

    linkaxes([ax1, ax2], 'xy');
    xticks(param.f_peak);
    xlim([1, 100]);
    max_val = max(Ht_avg(:));
    ylim([max_val/1000, max_val]); 
    
    set(findall(fig,'-property','FontSize'),'FontSize',26);
    saveas(fig, save_path);
    close(fig);
end