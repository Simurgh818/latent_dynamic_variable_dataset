function plotBandScatterPerTrial(Ht, Hr, f_plot, bands, band_names, param, k, methodName, save_dir)

nBands   = numel(band_names);
nHz      = size(Ht,1);
nLatents = size(Ht,2);

% -------- Marker + color per latent --------
markers = {'o','s','d','h','^','hexagram','<','>'};
assert(nLatents <= numel(markers), ...
    'Not enough markers defined for number of latents.');

colors = lines(nLatents);   % ← color per latent

% -------- Normalize amplitudes --------
Ht_amp = abs(Ht(1:nHz,:,:));
Hr_amp = abs(Hr(1:nHz,:,:));
Ht_amp = Ht_amp ./ max(Ht_amp(:));
Hr_amp = Hr_amp ./ max(Hr_amp(:));

true_vals  = cell(nBands,1);
recon_vals = cell(nBands,1);

for b = 1:nBands
    f_range  = bands.(band_names{b});
    idx_band = f_plot >= f_range(1) & f_plot <= f_range(2);

    % Mean over band → [latent × trial]
    true_vals{b}  = squeeze(mean(Ht_amp(idx_band,:,:),1,'omitnan'));
    recon_vals{b} = squeeze(mean(Hr_amp(idx_band,:,:),1,'omitnan'));
end

% -------- Plot --------
fig = figure('Position',[50 50 1500 320]);
tiledlayout(1,nBands,'TileSpacing','compact','Padding','compact');

sgtitle(sprintf('True vs %s Reconstructed FFT Band Amplitudes (k=%d)', ...
                methodName, k));

for b = 1:nBands
    nexttile; hold on;

    X = true_vals{b};
    Y = recon_vals{b};

    % ---- Scatter per latent ----
    for z = 1:nLatents
        scatter(X(z,:), Y(z,:), 22, ...
            'Marker', markers{z}, ...
            'MarkerEdgeColor', colors(z,:), ...
            'MarkerFaceColor', colors(z,:), ...
            'MarkerFaceAlpha', 0.25, ...
            'MarkerEdgeAlpha', 0.8, ...
            'DisplayName', sprintf('Z_{%d}', param.f_peak(z)));
    end

    % ---- y = x reference ----
    lim = [min([X(:);Y(:)]) max([X(:);Y(:)])];
    plot(lim, lim, 'k--', 'LineWidth',1.5, 'DisplayName','y = x');

    % ---- R² ----
    R = corrcoef(X(:), Y(:));
    if numel(R) > 1
        text(mean(lim), mean(lim), ...
            sprintf('R^2=%.2f', R(1,2)^2), ...
            'FontSize',12,'FontWeight','bold');
    end

    title([band_names{b} ' band']);

    if b == 1
        xlabel('True Band Amp.');
        ylabel('Recon. Band Amp.');
    end

    axis tight; grid on;
end
legend('Location','eastoutside', ...
               'Orientation','horizontal', ...
               'NumColumns', ceil((nLatents)/4), ...
               'TextColor','k');
set(findall(fig,'-property','FontSize'),'FontSize',15);

% -------- Save --------
if ~isempty(save_dir)
    saveas(fig, fullfile(save_dir, ...
        sprintf('%s_Scatter_BandAmp_Trials_k%d.png', methodName, k)));
end

end
