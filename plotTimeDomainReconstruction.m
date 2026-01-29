function plotTimeDomainReconstruction(h_true, h_recon, param, methodName, k, zeroLagCorr, save_dir)
% plotTimeDomainReconstruction Generates time-series comparison plots
%
% Inputs:
%   h_true      : True latent variables (Time x N_F)
%   h_recon     : Reconstructed latent variables (Time x N_F)
%   param       : Struct with .N_F, .f_peak, .fs
%   methodName  : String (e.g., 'ICA', 'PCA')
%   k           : Number of components used (integer)
%   zeroLagCorr : Vector of correlation values (1 x N_F)
%   save_dir    : Directory to save the PNG

    % --- 1. Parallel Safety Check ---
    % If running in a parfor worker, do nothing and return immediately.
    if ~isempty(getCurrentTask())
        return;
    end

    % --- 2. Setup ---
    num_vars = size(h_true, 2);
    h_f_colors = lines(num_vars);
    file_suffix = sprintf('_k%d', k);
    
    % Ensure f_peak is long enough to avoid crashes
    labels = param.f_peak;
    if length(labels) < num_vars
        % Pad with NaNs or placeholders if data has more vars than labels
        labels = [labels, nan(1, num_vars - length(labels))];
    end

    % --- 3. Create Figure ---
    % Invisible figure is faster and doesn't steal focus
    fig = figure('Position',[50 50 1200 150*num_vars], 'Visible', 'off');
    tiledlayout(num_vars, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    sgtitle([methodName ' (k=' num2str(k) ') Latent variables Z(t) and $\hat{z}(t)$'], ...
        'Interpreter','latex');
    
    % --- 4. Plot Loop ---
    for f = 1:num_vars
        nexttile;
        hold on;
        set(gca, 'XColor', 'none', 'YColor', 'none'); 
        box on;
        
        % Safe Label Generation
        if isnan(labels(f))
            lbl_true  = sprintf('$Z_{%d}$ (t)', f);
            lbl_recon = sprintf('$\\hat{Z}_{%d}$ (t)', f);
        else
            lbl_true  = ['$Z_{' num2str(labels(f)) '}$ (t) '];
            lbl_recon = ['$\hat{Z}_{' num2str(labels(f)) '}$ (t) '];
        end

        % Plot True
        plot(h_true(:, f), 'LineStyle', '-', 'Color', h_f_colors(f, :), ...
            'DisplayName', lbl_true);
            
        % Plot Recon
        plot(h_recon(:, f), 'LineStyle', '--', 'Color', 'k', ...
            'DisplayName', lbl_recon);
        
        ylabel('Amp');
        % Limit to 2 seconds or full length, whichever is shorter
        max_time = min(size(h_true,1), param.fs);
        xlim([0 max_time]); 
        
        legend('Show', 'Interpreter', 'latex', 'Location', 'eastoutside');
        
        % Add Correlation Text
        if f <= length(zeroLagCorr)
            rho = zeroLagCorr(f);
            % Position text relative to the max amplitude of the current trace
            y_pos = 0.7 * max(abs(h_true(:, f)));
            text(0.02 * param.fs, y_pos, ...
                sprintf('\\rho(0)=%.2f', rho), ...
                'FontSize', 12, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], 'BackgroundColor', 'w', ...
                'Margin', 3, 'EdgeColor', 'k');
        end
        hold off;
    end
    
    % --- 5. Scale Bars (On the last tile) ---
    % Note: Assumes normalized data (std=1). If amplitude varies wildly, 
    % fixed offsets (y0+2) might need adjustment.
    x0 = 0; 
    y0 = min(ylim) + 0.2;
    
    % X-bar (1 sec)
    line([x0 x0+param.fs/2], [y0 y0], 'Color', 'k', 'LineWidth', 2, 'HandleVisibility', 'off');
    text(x0+param.fs/2, y0-0.1, '500 msec', 'VerticalAlignment', 'top');
    
    % Y-bar (2 units)
    line([x0 x0], [y0 y0+2], 'Color', 'k', 'LineWidth', 2, 'HandleVisibility', 'off');
    text(x0-5, y0+2.5, '2 a.u.', 'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'right', 'Rotation', 90);
    
    % --- 6. Save and Close ---
    set(findall(fig, '-property', 'FontSize'), 'FontSize', 16);
    
    filename = sprintf('%s_TimeDomain%s.png', methodName, file_suffix);
    saveas(fig, fullfile(save_dir, filename));
    close(fig);
end