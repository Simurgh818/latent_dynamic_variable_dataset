function plotTimeDomainReconstruction(h_true, h_recon, param, methodName, k, Corr, save_dir)
% plotTimeDomainReconstruction Generates stacked time-series comparison plots
% with Okabe-Ito colorblind-safe colors, scaled amplitudes, +4pt fonts, and 600 DPI export.

    % --- 1. Parallel Safety Check ---
    if ~isempty(getCurrentTask())
        return;
    end

    % --- 2. Setup & Okabe-Ito Palette (Matches Panel C) ---
    num_vars = size(h_true, 2);
    cb_palette = [
        0.00, 0.45, 0.70;  % Blue           (Z_1 / 2 Hz)
        0.90, 0.60, 0.00;  % Orange         (Z_2 / 6 Hz)
        0.00, 0.60, 0.50;  % Bluish Green   (Z_3 / 10 Hz)
        0.80, 0.40, 0.00;  % Vermillion     (Z_4 / 18 Hz)
        0.35, 0.70, 0.90;  % Sky Blue       (Z_5 / 24 Hz)
        0.80, 0.60, 0.70;  % Reddish Purple (Z_6 / 50 Hz)
        0.95, 0.90, 0.25;  % Yellow (Fallback)
        0.20, 0.20, 0.20   % Dark Gray (Fallback)
    ];
    h_f_colors = cb_palette(1:num_vars, :);
    file_suffix = sprintf('_k%d', k);
    
    % Crop to 1 second (1000 ms)
    max_samples = min(size(h_true, 1), round(param.fs));
    t_ms = (0:max_samples-1) / param.fs * 1000;

    % --- SPACING AND SCALING (PREVENTS OVERLAP) ---
    offset    = 4.0;   % Vertical spacing between latent baselines
    amp_scale = 0.55;  % Scale factor to keep peaks within each latent's lane

    % --- 3. Create Figure on a Single Axis ---
    fig = figure('Position', [50, 50, 1000, 750], 'Visible', 'off');
    hold on;
    box on;

    % Set base axis properties FIRST so they don't override title/label sizes
    set(gca, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20);
    grid on;

    % Plot from top (Z_1) to bottom (Z_num_vars)
    for f = 1:num_vars
        y_shift = (num_vars - f) * offset;

        % 1. True Latent (Solid Okabe-Ito colored line, LineWidth 2.5)
        plot(t_ms, (h_true(1:max_samples, f) * amp_scale) + y_shift, ...
            'LineStyle', '-', 'Color', h_f_colors(f, :), ...
            'LineWidth', 2.5);
            
        % 2. Reconstructed Latent (Dashed dark line, LineWidth 2.0)
        plot(t_ms, (h_recon(1:max_samples, f) * amp_scale) + y_shift, ...
            'LineStyle', '--', 'Color', [0.15 0.15 0.15], ...
            'LineWidth', 2.0);
        
        % 3. Correlation Box
        if f <= length(Corr)
            rho = Corr(f);
            text(25, y_shift + 1.35, sprintf('\\rho=%.2f', rho), ...
                'FontSize', 17, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
                'Margin', 3, 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 1.2);
        end
    end

    % --- 4. Y-Axis Formatting ---
    tick_positions = (0 : num_vars - 1) * offset;
    yticks(tick_positions);
    
    y_labels = arrayfun(@(i) sprintf('Z_{%d}', i), num_vars:-1:1, 'UniformOutput', false);
    yticklabels(y_labels);
    
    ylabel('Z-Scored Amplitude (stacked)', 'FontSize', 22, 'FontWeight', 'bold');
    ylim([-offset * 0.7, (num_vars - 0.05) * offset]);

    % --- 5. X-Axis Formatting ---
    xlim([0, 1000]);
    xticks(0:200:1000);
    xlabel('Time (ms)', 'FontSize', 22, 'FontWeight', 'bold');

    % --- 6. Title and Legend (Uses supported cmbx10 font - No SceneNode warnings) ---
    title(sprintf('\\textbf{%s (k=%d) Latent variables} $\\;\\mathbf{Z(t)}\\;$ \\textbf{and} $\\;\\mathbf{\\hat{z}(t)}$', methodName, k), ...
        'Interpreter', 'latex', 'FontSize', 26);

    % Clean 2-entry legend using Okabe-Ito Blue for Z(t) True
    h_leg_true  = plot(nan, nan, '-', 'Color', cb_palette(1, :), 'LineWidth', 2.5);
    h_leg_recon = plot(nan, nan, '--', 'Color', [0.15 0.15 0.15], 'LineWidth', 2.0);
    legend([h_leg_true, h_leg_recon], {'\textbf{Z(t) True}', '$\mathbf{\hat{z}(t)}$ \textbf{Recon}'}, ...
        'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 17);

    hold off;

    % --- 7. Save at 600 DPI and Close ---
    filename = sprintf('%s_TimeDomain%s.png', methodName, file_suffix);
    exportgraphics(fig, fullfile(save_dir, filename), 'Resolution', 600);
    close(fig);
end