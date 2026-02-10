function plotCumulativeVariance(explained, k_sig, method_name, save_path)
    % explained: vector from pca function
    % k_sig: the specific k being evaluated (for the red dashed line)
    
    fig = figure('Position', [100, 100, 800, 400], 'Visible', 'off');
    hold on;
    
    cum_var = cumsum(explained);
    plot(cum_var, 'o-', 'LineWidth', 2, 'MarkerSize', 4, 'Color', [0.2 0.4 0.7]);
    
    % Highlight the current k
    xline(k_sig, '--r', 'LineWidth', 1.5, 'DisplayName', sprintf('k=%d', k_sig));
    
    % Formatting
    grid on;
    xlabel('Number of Components');
    ylabel('Cumulative Variance Explained (%)');
    title(sprintf('%s: Explained Variance', method_name));
    ylim([0 105]);
    set(gca, 'FontSize', 14);
    
    saveas(fig, save_path);
    close(fig);
end