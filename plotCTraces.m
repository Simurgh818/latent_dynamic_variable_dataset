function plotCTraces(num_sig_components, param, components_test, h_true, corr_table, method_dir, file_suffix)
    % plotCTraces: Creates a stacked trace plot overlaying true latents with matched raw components
    % using the Okabe-Ito colorblind-safe palette and 600 DPI export.
    
    % Determine how many latents to plot 
    num_comps_plot = min([param.N_F, size(h_true, 2), height(corr_table)]);
    
    % Define time vector for the first 1 second (1000 ms)
    t_plot_sec = 1; 
    num_samples = min([size(components_test, 1), size(h_true, 1), round(t_plot_sec * param.fs)]);
    time_vec = (0:num_samples-1) / param.fs * 1000;
    
    % --- 1. Setup Figure (1000 x 750) ---
    fig2 = figure('Position', [50, 50, 1000, 750], 'Visible', 'off');
    hold on;
    box on;
    
    % --- Okabe-Ito Colorblind-Safe Palette (Matches Panel C) ---
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
    h_f_colors = cb_palette(1:max(param.N_F, num_comps_plot), :);
    
    % Fixed offset for Z-scored data to keep traces visually separated
    offset = 10; 
    
    for i = 1:num_comps_plot
        % 1. Extract the matching pair from the table
        f = corr_table.h_f(i);
        
        if ismember('C', corr_table.Properties.VariableNames)
            c = corr_table.C(i);
        elseif ismember('Component', corr_table.Properties.VariableNames)
            c = corr_table.Component(i);
        else
            c = corr_table{i, 1}; % Fallback 
        end
        
        corr_val = corr_table.corr_value(i);
        
        % Ensure the component exists (safeguard for k < 6)
        if c > size(components_test, 2)
            continue;
        end
        
        % 2. Extract Data
        trace_true  = h_true(1:num_samples, f);
        trace_recon = components_test(1:num_samples, c);
        
        % 3. Z-Score Normalize both so they share the same amplitude scale
        trace_true_z  = zscore(trace_true);
        trace_recon_z = zscore(trace_recon);
        
        % 4. Flip the sign of the component if negatively correlated
        if corr_val < 0
            trace_recon_z = trace_recon_z * -1;
        end
        
        % Calculate vertical shift (Latent 1 at the top)
        y_shift = (num_comps_plot - i) * offset;
        
        % Plot 1: True Latent (Solid Okabe-Ito colored line, LineWidth 2.5)
        plot(time_vec, trace_true_z + y_shift, 'Color', h_f_colors(f, :), ...
             'LineWidth', 2.5, 'DisplayName', sprintf('True Z_{%d}', f));
             
        % Plot 2: Matched Raw Component (Dashed dark line, LineWidth 2.0)
        plot(time_vec, trace_recon_z + y_shift, 'Color', [0.15 0.15 0.15], 'LineStyle', '--', ...
             'LineWidth', 2.0, 'DisplayName', sprintf('Comp_{%d}', c));
    end
    
    hold off;
    
    % --- 2. X and Y Axes Formatting (+4pt Font Increase) ---
    set(gca, 'Box', 'on', 'LineWidth', 1.5, 'FontSize', 20);
    grid on;
    
    xlim([0, 1000]);
    xticks(0:200:1000);
    ylim([-offset * 0.7, num_comps_plot * offset]);
    
    yticks((0:num_comps_plot-1) * offset);
    yticklabels(arrayfun(@(idx) sprintf('Z_{%d}', corr_table.h_f(num_comps_plot - idx + 1)), ...
        1:num_comps_plot, 'UniformOutput', false));
    
    xlabel('Time (msec)', 'FontSize', 22, 'FontWeight', 'bold');
    ylabel('Z-Scored Amplitude (stacked)', 'FontSize', 22, 'FontWeight', 'bold');
    title(sprintf('Matched Latents vs Raw Components (k=%d)', num_sig_components), ...
          'FontSize', 26, 'FontWeight', 'bold');
    
    % --- 3. Save at 600 DPI ---
    filename = sprintf('Component_Traces_Stacked%s.png', file_suffix);
    exportgraphics(fig2, fullfile(method_dir, filename), 'Resolution', 600);
    close(fig2);
end