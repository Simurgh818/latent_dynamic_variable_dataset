clc; clear;
% Simulated EEG set

% This ensures rand() and randn() produce the same sequence every time.
rng(42,'twister');

% freq_peak_latents = [2 2.4 8 20 21 32 40 40];
% freq_peak_latents = [40 40 32 21 20 8 2.4 2];
freq_peak_latents = [2 5 8 20 21 32 40 40];

num_latents = length(freq_peak_latents);
zeta_latents = 0.15;
T = 1000;     % Duration, in seconds
dt = 0.005;   % time step, in seconds
all_h_F = zeros(length(freq_peak_latents), T/dt);

for i_fpl= 1:length(freq_peak_latents)
    h_F = generateSDHO(freq_peak_latents(i_fpl), zeta_latents, dt, T);
    all_h_F(i_fpl, :) = h_F;
end
% normalize temporal latents (unit variance)
all_h_F = all_h_F ./ std(all_h_F, [], 2);

%% Set up the Import Options and import the data
% Load EEG electrode locations
opts = delimitedTextImportOptions("NumVariables", 2);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["x", "y"];
opts.VariableTypes = ["double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
approxeeglocs = readtable("C:/Users/sdabiri/OneDrive - Georgia Institute of Technology/Dr. Sederberg MaTRIX Lab/Shared Code/latent_dynamic_variable_dataset/approx_eeg_locs.csv", opts);

% Clear temporary variables
clear opts
% Model is spatial filter + tanh nonlinearity 
eeg_loc_x = approxeeglocs.x;
eeg_loc_y = approxeeglocs.y;
num_channels = length(eeg_loc_x);
%% get full component images 
num_spatial_realizations = 10;

for i_spat = 1:num_spatial_realizations

    % get postive source locs
    pos_src_locs = rand(num_latents, 2)*1.5 - 0.75;
    neg_src_locs = rand(num_latents, 2)*1.5 - 0.75;
    
    src_widths = 0.5 + 0.05*randn(num_latents, 2);
    src_pks = 1 + 0.1*rand(num_latents, 2);
    
    figure()
    plot(eeg_loc_x, eeg_loc_y, 'ko')
    hold on
    % Replace line 53 with:
    plot(pos_src_locs(:, 1), pos_src_locs(:, 2), 'r*')
    hold on
    plot(neg_src_locs(:, 1), neg_src_locs(:, 2), 'b*') % Plot negative sources in blue
    legend('EEG Locs', 'Pos Sources', 'Neg Sources')
    
    % Continuous spatial masks
    [mesh_x, mesh_y] = meshgrid(-.8:0.05:0.8);
    all_comp_masks = repmat(mesh_x, [1 1 num_latents]);
    figure()
    for i_fpl = 1:num_latents
        comp_mask = spatial_mask_fun(pos_src_locs(i_fpl, :), neg_src_locs(i_fpl, :), ...
            src_widths(i_fpl, :), src_pks(i_fpl,:), mesh_x, mesh_y);
        all_comp_masks(:, :, i_fpl) = comp_mask;
        
        nexttile
        imagesc(comp_mask)
    end
    save(sprintf('source_params%02d_key.mat', i_spat), 'src_pks', 'src_widths', 'eeg_loc_y', 'eeg_loc_x', ...
        'all_comp_masks', 'freq_peak_latents', 'zeta_latents')
    
    % now sample spatial filters at EEG locations
    
    spatial_comps = zeros(num_channels, num_latents);
    for i_fpl = 1:num_latents
        spatial_comps(:, i_fpl) = interp2(mesh_x, mesh_y, all_comp_masks(:, :, i_fpl), eeg_loc_x, eeg_loc_y);
        % normalize spatial components
        spatial_comps(:, i_fpl) = spatial_comps(:, i_fpl) / norm(spatial_comps(:, i_fpl));
    end
    
    % check that it worked
    
    figure()
    nexttile
    imagesc(mesh_x(1, :), mesh_y(:, 1), all_comp_masks(:, :, 5))
    set(gca, 'YDir', 'normal')
    hold on
    scatter(eeg_loc_x, eeg_loc_y, 100, (spatial_comps(:, 5)), 'filled', 'MarkerEdgeColor', [0 0 0])
    axis equal tight
    title('should match')
    nexttile
    imagesc(mesh_x(1, :), mesh_y(:, 1), all_comp_masks(:, :, 5))
    hold on
    scatter(eeg_loc_x, eeg_loc_y, 100, (spatial_comps(:, 3)), 'filled', 'MarkerEdgeColor', [0 0 0])
    title('shouldn''t match')
    %  Great, now use tanh + spatial filter to generate EEG
    % select_comps = [1 2 4 7];
    % wx_vals = spatial_comps(:, select_comps)*all_h_F(select_comps, :);
    % 
    % % small gain, bias 0 : approximately linear
    % gain_par = 0.2;
    % bias_par = 0;
    % sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);
    % 
    % train_t_range = 1:120000;
    % test_t_range = 150000:200000;
    % train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    % test_sim_eeg_vals = sim_eeg_vals(:, 150000:end);
    % train_true_hF = all_h_F(:, train_t_range);
    % test_true_hF = all_h_F(:, test_t_range);
    % 
    % save('exploratory_scripts/simEEG_set1_randF.mat', "train_sim_eeg_vals", ...
    %     "train_true_hF", "test_sim_eeg_vals", "dt")
    % 
    % save('exploratory_scripts/simEEG_set1_key_randF.mat', "test_sim_eeg_vals", "test_true_hF", ...
    %     "select_comps", "spatial_comps", "gain_par", "bias_par")
    
    % set 2: still linear, more components
    select_comps = 1:8;
    wx_vals = spatial_comps(:, select_comps)*all_h_F(select_comps, :);
    
    % small gain, bias 0 : approximately linear
    gain_par = 0.2;
    bias_par = 0;
    sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);
    
    train_t_range = 1:120000;
    test_t_range = 150000:200000;
    train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    test_sim_eeg_vals = sim_eeg_vals(:, 150000:end);
    train_true_hF = all_h_F(:, train_t_range);
    test_true_hF = all_h_F(:, test_t_range);
    
    save(sprintf('simEEG_set2_spat%02d.mat', i_spat), "train_sim_eeg_vals", ...
        "train_true_hF", "test_sim_eeg_vals", "dt")
    
    save(sprintf('simEEG_set2_spat%02d_key.mat', i_spat), "test_sim_eeg_vals", "test_true_hF", ...
        "pos_src_locs", "neg_src_locs", "src_widths", "src_pks",...
        "select_comps", "spatial_comps", "gain_par", "bias_par")
    
    % Set 3: nonlinear
    % select_comps = [1 2 4 7];
    % wx_vals = spatial_comps(:, select_comps)*all_h_F(select_comps, :);
    % 
    % % small gain, bias 0 : approximately linear
    % gain_par = 2;
    % bias_par = 1;
    % sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);
    % 
    % train_t_range = 1:120000;
    % test_t_range = 150000:200000;
    % train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    % test_sim_eeg_vals = sim_eeg_vals(:, 150000:end);
    % train_true_hF = all_h_F(:, train_t_range);
    % test_true_hF = all_h_F(:, test_t_range);
    % 
    % save('exploratory_scripts/simEEG_set3.mat', "train_sim_eeg_vals", ...
    %     "train_true_hF", "test_sim_eeg_vals", "dt")
    % 
    % save('exploratory_scripts/simEEG_set3_key_.mat', "test_sim_eeg_vals", "test_true_hF", ...
    %     "select_comps", "spatial_comps", "gain_par", "bias_par")
    
    % Set 4: nonlinear, all components
    select_comps = 1:8;
    wx_vals = spatial_comps(:, select_comps)*all_h_F(select_comps, :);
    
    % small gain, bias 0 : approximately linear
    gain_par = 2;
    bias_par = 1;
    sim_eeg_vals = tanh(gain_par*wx_vals + bias_par);
    
    train_t_range = 1:120000;
    test_t_range = 150000:200000;
    train_sim_eeg_vals = sim_eeg_vals(:, train_t_range);
    test_sim_eeg_vals = sim_eeg_vals(:, 150000:end);
    train_true_hF = all_h_F(:, train_t_range);
    test_true_hF = all_h_F(:, test_t_range);
    
    save(sprintf('simEEG_set4_spat%02d.mat', i_spat), "train_sim_eeg_vals", ...
        "train_true_hF", "test_sim_eeg_vals", "dt")
    
    save(sprintf('simEEG_set4_spat%02d_key.mat', i_spat), "test_sim_eeg_vals", "test_true_hF", ...
        "pos_src_locs", "neg_src_locs", "src_widths", "src_pks", ...
        "select_comps", "spatial_comps", "gain_par", "bias_par")

end
%% Parsimonious plots: just show the electrodes, color by component
t_range = 1:500;
makeMyFigure(20, 15);

tiledlayout(4, 6)
for i_comp = 1:size(spatial_comps, 2)
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 30, spatial_comps(:, i_comp), 'filled')
    title(['s_{i' num2str(i_comp) '}'], 'interpreter','tex')
    axis tight
    axis equal
    set(gca, 'box', 'off', 'color', 'none', 'xtick', [], 'ytick', [])
% set(gca, 'visible', 'off')
    nexttile([1 2])
    plot(dt*t_range, test_true_hF(i_comp, t_range), 'k')
    axis tight
    set(gca, 'color', 'none', 'box', 'off')
    ylabel(['h_' num2str(i_comp) '(t)'], 'Interpreter','tex')
    title(['f_{peak} = ' num2str(freq_peak_latents(i_comp), '%1.2f') ' Hz'], 'Interpreter','tex' )
end

exportgraphics(gcf, 'layout1_fig.pdf')
savefig('layout1_fig.fig')
%%
t_range = 1:1000;
figure()
nexttile
hold on
for i_hf = select_comps
    plot(dt*t_range, all_h_F(i_hf, t_range)+i_hf)
end
%%
figure()
nexttile
hold on
for i_ch = 1:num_channels
    plot(dt*t_range, sim_eeg_vals(i_ch, t_range) + i_ch)
end

%% pca try (hopefully it fails...)

[coeff, score, latent, tsquared, explained] = pca(train_sim_eeg_vals');

figure()
tiledlayout(6, 9)
for i_c = 1:10
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 100, coeff(:, i_c), 'filled', 'MarkerEdgeColor', [0 0 0])
    title(['pca component' num2str(i_c)])

    nexttile([1 2])
    plot(dt*t_range', score(t_range, i_c))
    axis tight
    set(gca, 'visible', 'off')
end

for i_lc = select_comps
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 100, (spatial_comps(:, i_lc)), 'filled', 'MarkerEdgeColor', [0 0 0])
    title(['true component ' num2str(i_lc)])

    nexttile([1 2])
    plot(dt*t_range, train_true_hF(i_lc, t_range))
    axis tight
    set(gca, 'visible', 'off')
end


%% autoencoder try

% train_range = 1:100000;
% test_range = 100001:200000;
[train_x_zscore, mu_x, sig_x] = zscore(train_sim_eeg_vals, [], 2);

% try stacking a few timepoints
max_skip = 50;
skip1 = 2;
skip2 = 3;
skip3 = 4;
skip4 = 10;
skip5 = 20;
stack_train_x_zscore = [train_x_zscore(:, 1:end - max_skip); ...
    train_x_zscore(:, skip1:end - max_skip+skip1-1); ...
    train_x_zscore(:, skip2:end - max_skip + skip2 - 1); ...
    train_x_zscore(:, skip3:end - max_skip + skip3 - 1); ...
    train_x_zscore(:, skip4:end - max_skip + skip4 - 1); ...
    train_x_zscore(:, skip5:end - max_skip + skip5 - 1)];

% train_h_f = all_h_F(:, train_range);
% test_h_f = all_h_F(:, test_range);

hiddenSize = 15;
autoenc = trainAutoencoder(stack_train_x_zscore(:, 1:100:end), hiddenSize, ...
    'MaxEpochs', 200, 'EncoderTransferFunction', 'logsig', ...
    'DecoderTransferFunction', 'logsig', 'L2WeightRegularization', 0, ...
    'SparsityRegularization', 0, 'SparsityProportion', 0.5);
%% train a neural network to match encoded Z(t) to h_F(t)


% z_train = encode(autoenc, train_x_zscore);
z_train = encode(autoenc, stack_train_x_zscore);

%% is z_train correlated with h_F? 

% hz_cc = corrcoef([z_train; train_h_f]');
hz_cc = corrcoef([z_train; train_true_hF(:, 1:end-max_skip)]');

figure()
imagesc(hz_cc)
%%
figure()
tiledlayout(9, 9)
aenc_w = autoenc.EncoderWeights;
for i_c = 1:15
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 100, aenc_w(i_c, 1:31), 'filled', 'MarkerEdgeColor', [0 0 0])
    title(['AE component' num2str(i_c)])

    nexttile([1 2])
    plot(dt*t_range', z_train(i_c, t_range))
    axis tight
    set(gca, 'visible', 'off')
end

for i_lc = select_comps
    nexttile
    scatter(eeg_loc_x, eeg_loc_y, 100, (spatial_comps(:, i_lc)), 'filled', 'MarkerEdgeColor', [0 0 0])
    title(['true component ' num2str(i_lc)])

    nexttile([1 2])
    plot(dt*t_range, train_true_hF(i_lc, t_range))
    axis tight
    set(gca, 'visible', 'off')
end
%%
hz_net = feedforwardnet(10, 'trainlm');
hz_net = train(hz_net, z_train(:, 1:20:end), train_h_f(:, 1:20:end-max_skip));

pred_h_train = hz_net(z_train);
%%
figure()
for i_c = 1:num_latents
nexttile
plot(pred_h_train(i_c, :)', train_h_f(i_c, 1:end-max_skip)', 'o')
eqline  
end
%%
test_x_zscore = (test_x - mu_x)./sig_x;
xReconstructed = predict(autoenc, test_x_zscore);
mseError = mse(test_x_zscore-xReconstructed)

z_test = encode(autoenc, test_x_zscore);
recon_test = decode(autoenc, z_test);

recon_test_orig = diag(sig_x)*recon_test + mu_x;


pred_h_test = hz_net(z_test);

figure()
plot(pred_h_test', test_h_f', 'o')
eqline


%%
% make masks for peaks
function comp_mask = spatial_mask_fun(pos_src_loc, neg_src_loc, src_widths, src_pks, mesh_x, mesh_y)

pos_comp = src_pks(1)*exp(-((mesh_x - pos_src_loc(1)).^2 + (mesh_y - pos_src_loc(2)).^2)/(2*src_widths(1)^2));
neg_comp = src_pks(2)*exp(-((mesh_x - neg_src_loc(1)).^2 + (mesh_y - neg_src_loc(2)).^2)/(2*src_widths(1)^2));

comp_mask = pos_comp - neg_comp;
end