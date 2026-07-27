function plotCTraces(num_sig_components, param, components_test, h_true, corr_table, method_dir, file_suffix)
    % plotCTraces: Creates a stacked trace plot overlaying true latents with matched raw components
    %
    % Inputs:
    %   num_sig_components : Total components in the run (k)
    %   param              : Struct containing .fs and .N_F
    %   components_test    : Raw extracted components (Time x k)
    %   h_true             : True latent fields (Time x Latents)
    %   corr_table         : Table matching components to latents
    %   method_dir         : Output directory
    %   file_suffix        : Suffix for filename
    
    % Determine how many latents to plot 
    num_comps_plot = min([param.N_F, size(h_true, 2), height(corr_table)]);
    
    % Define time vector for the first 1 second
    t_plot_sec = 1; 
    num_samples = min([size(components_test, 1), size(h_true, 1), round(t_plot_sec * param.fs)]);
    time_vec = (0:num_samples-1) / param.fs * 1000;
    
    % Setup Figure
    fig2 = figure('Position', [50 50 1200 900], 'Visible', 'off');
    hold on;
    
    % Generate the color palette for the true latents
    h_f_colors = lines(param.N_F);
    
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
        
        % 4. Flip the sign of the component if they are negatively correlated
        if corr_val < 0
            trace_recon_z = trace_recon_z * -1;
        end
        
        % Calculate vertical shift (Latent 1 at the top)
        y_shift = (num_comps_plot - i) * offset;
        
        % Plot 1: True Latent (Thick, colored line)
        plot(time_vec, trace_true_z + y_shift, 'Color', h_f_colors(f, :), ...
             'LineWidth', 2.5, 'DisplayName', sprintf('True Z_{%d}', f));
             
        % Plot 2: Matched Raw Component (Thinner, black dashed line)
        plot(time_vec, trace_recon_z + y_shift, 'Color', 'k', 'LineStyle', '--', ...
             'LineWidth', 1.5, 'DisplayName', sprintf('Comp_{%d}', c));
    end
    
    hold off;
    
    % --- Formatting ---
    xlim([0 t_plot_sec*1000]);
    ylim([-offset, num_comps_plot * offset]);
    
    xlabel('Time (msec)');
    ylabel('Z-Scored Amplitude (stacked)');
    title(['Matched Latents vs Raw Components (k=' num2str(num_sig_components) ')']);
    
    % Set ticks to show Z (Latent) numbers clearly
    yticks((0:num_comps_plot-1) * offset);
    yticklabels(arrayfun(@(idx) sprintf('Z_{%d}', corr_table.h_f(num_comps_plot - idx + 1)), ...
        1:num_comps_plot, 'UniformOutput', false));
    
    grid on;
    set(findall(fig2, '-property', 'FontSize'), 'FontSize', 22);
    
    % Save
    saveas(fig2, fullfile(method_dir, ['Component_Traces_Stacked' file_suffix '.png']));
    close(fig2);
end