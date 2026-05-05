function entry = runDimRedMethod( ...
    method, data, local_param, k, ki, condition, dataset_name, ...
    method_dir, local_results_dir)

% Initialize generic temporary variables
R2  = NaN;
MSE = NaN;
out = struct();
corr_table = table(); % default empty
Comp_latent_matching_matrix = [];
spectral_R2 = NaN; % Initialize spectral_R2 for cases where it's not defined

% Run Analysis
switch method
    case 'PCA'
        [outPCA] = runPCAAnalysis(data.eeg_train, data.eeg_test,...
            data.H_train, data.H_test, local_param, k, method_dir);
                        
        out = outPCA;
        if isfield(outPCA, 'corr_PCA'), corr_table = outPCA.corr_PCA; end
        if isfield(outPCA, 'Comp_latent_matching_matrix'),   Comp_latent_matching_matrix   = outPCA.Comp_latent_matching_matrix;   end
        if isfield(outPCA, 'spectral_R2'), spectral_R2 = outPCA.spectral_R2; end
        if isfield(outPCA, 'avg_comp_corr'), avg_comp_corr = outPCA.avg_comp_corr; end

   case 'dPCA'
        % Assuming rundPCAAnalysis also takes 'k' now and returns single-k result
        [outDPCA] = rundPCAAnalysis( ...
            data.eeg_train, data.eeg_test, data.H_train, data.H_test,...
            local_param, k, method_dir);       
        
        out = outDPCA;
        if isfield(outDPCA, 'corr_dPCA'), corr_table = outDPCA.corr_dPCA; end
        if isfield(outDPCA, 'Comp_latent_matching_matrix'),    Comp_latent_matching_matrix   = outDPCA.Comp_latent_matching_matrix;    end
        if isfield(outDPCA, 'spectral_R2'), spectral_R2 = outDPCA.spectral_R2; end
        if isfield(outDPCA, 'avg_comp_corr'), avg_comp_corr = outDPCA.avg_comp_corr; end

   case 'ICA'
        [outICA] = runICAAnalysis(data.eeg_train, data.eeg_test, ...
             data.H_train, data.H_test, k, local_param, method_dir);
        out = outICA;
        if isfield(outICA, 'corr_ICA'), corr_table = outICA.corr_ICA; end
        if isfield(outICA, 'Comp_latent_matching_matrix'),   Comp_latent_matching_matrix   = outICA.Comp_latent_matching_matrix;   end
        if isfield(outICA, 'spectral_R2'), spectral_R2 = outICA.spectral_R2; end
        if isfield(outICA, 'avg_comp_corr'), avg_comp_corr = outICA.avg_comp_corr; end
        
    case 'UMAP'
        n_neighbors = 3; min_dist = 0.99;
        [outUMAP] = runUMAPAnalysis( ...
            n_neighbors, min_dist, data.eeg_train, data.eeg_test, local_param, ...
            data.H_train, data.H_test, k, local_results_dir);
        out = outUMAP;
        if isfield(outUMAP, 'corr_UMAP'), corr_table = outUMAP.corr_UMAP; end
        if isfield(outUMAP, 'Comp_latent_matching_matrix'),    Comp_latent_matching_matrix   = outUMAP.Comp_latent_matching_matrix;    end
        if isfield(outUMAP, 'spectral_R2'), spectral_R2 = outUMAP.spectral_R2; end
        if isfield(outUMAP, 'avg_comp_corr'), avg_comp_corr = outUMAP.avg_comp_corr; end

    case 'AE'
        [outAE] = runAutoencoderAnalysis(data.eeg_train, data.eeg_test,...
            data.H_train, data.H_test, k, local_param, local_results_dir);
        out = outAE;
        if isfield(outAE, 'corr_AE'), corr_table = outAE.corr_AE; end
        if isfield(outAE, 'Comp_latent_matching_matrix'),  Comp_latent_matching_matrix   = outAE.Comp_latent_matching_matrix;  end
        if isfield(outAE, 'spectral_R2'), spectral_R2 = outAE.spectral_R2; end
        if isfield(outAE, 'avg_comp_corr'), avg_comp_corr = outAE.avg_comp_corr; end

    case 'iVAE'
        % --- Define your Beta value ---
        beta_val = 0.1; 
        [outIVAE] = runIVaeAnalysis(data.eeg_train, data.c_train, ...
                                             data.eeg_test, data.c_test, ...
                                             data.H_train, data.H_test, ...
                                             k, local_param, local_results_dir, beta_val);

        out = outIVAE;

        if isfield(outIVAE, 'corr_IVAE'),   corr_table  = outIVAE.corr_IVAE;   end
        if isfield(outIVAE, 'Comp_latent_matching_matrix'),      Comp_latent_matching_matrix    = outIVAE.Comp_latent_matching_matrix;      end
        if isfield(outIVAE, 'spectral_R2'), spectral_R2 = outIVAE.spectral_R2; end
        if isfield(outIVAE, 'avg_comp_corr'), avg_comp_corr = outIVAE.avg_comp_corr; end
end

% Process Correlation Table Metadata
if istable(corr_table) && ~isempty(corr_table)
    corr_table.method  = repmat(string(method), height(corr_table), 1);
    corr_table.dataset = repmat(string(dataset_name), height(corr_table), 1);
    corr_table.k       = repmat(k, height(corr_table), 1);
end

entry = struct();
entry.condition = string(condition);
entry.dataset = string(dataset_name);
entry.method  = string(method);
entry.k       = k;
entry.ki      = ki;
entry.stats = struct();
entry.direct_Component_Corr = corr_table;
entry.Comp_latent_matching_matrix = Comp_latent_matching_matrix;
entry.avg_comp_corr = avg_comp_corr;
entry.spectral_R2 = spectral_R2;
entry.out  = out;

end