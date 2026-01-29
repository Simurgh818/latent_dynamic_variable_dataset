function plotCTraces(num_sig_components, param, score, method_dir, file_suffix)

if num_sig_components <= param.N_F
    num_comps_plot = num_sig_components;
else
    num_comps_plot = param.N_F;
end

fig2 = figure('Position',[50 50 1000 (num_comps_plot*300)/2]);
tiledlayout(num_comps_plot, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle(['Component Traces (k=' num2str(num_sig_components) ')']);
for pc=1:num_comps_plot
    nexttile;
    plot(score(:,pc), 'LineStyle', '-', 'Color', 'k','DisplayName', ['C_' num2str(pc) '(t)']);
    xlabel('Time (msec)'); ylabel('Amp.');
    xlim([0 param.fs * 2]);
    legend('show');
end
set(findall(fig2,'-property','FontSize'),'FontSize',16);
saveas(fig2, fullfile(method_dir, ['Component_Traces_' file_suffix '.png']));

end