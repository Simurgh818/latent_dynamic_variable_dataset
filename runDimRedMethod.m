function entry = runDimRedMethod( ...
    method, data, local_param, k, ki, condition, dataset_name, ...
    method_dir, local_results_dir)

% Initialize generic temporary variables
R2  = NaN;
MSE = NaN;
out = struct();
corr_table = table(); % default empty
Comp_latent_matching_matrix = [];
spectral_R2 = NaN; 
direct_Component_Corr = NaN; % <--- Initialized to prevent errors if a method fails

% Run Analysis
switch method
    case 'PCA'
        [outPCA] = runPCAAnalysis(data.eeg_train, data.eeg_test,...
            data.H_train, data.H_test, local_param, k, method_dir);
                        
        out = outPCA;
        if isfield(outPCA, 'Comp_latent_matching_corr'), corr_table = outPCA.Comp_latent_matching_corr; end
        if isfield(outPCA, 'Comp_latent_matching_matrix'),   Comp_latent_matching_matrix   = outPCA.Comp_latent_matching_matrix;   end
        if isfield(outPCA, 'spectral_R2'), spectral_R2 = outPCA.spectral_R2; end
        if isfield(outPCA, 'direct_Component_Corr'), direct_Component_Corr = outPCA.direct_Component_Corr; end
        
   case 'dPCA'
        [outDPCA] = rundPCAAnalysis( ...
            data.eeg_train, data.eeg_test, data.H_train, data.H_test,...
            local_param, k, method_dir);       
        
        out = outDPCA;
        if isfield(outDPCA, 'Comp_latent_matching_corr'), corr_table = outDPCA.Comp_latent_matching_corr; end
        if isfield(outDPCA, 'Comp_latent_matching_matrix'),    Comp_latent_matching_matrix   = outDPCA.Comp_latent_matching_matrix;    end
        if isfield(outDPCA, 'spectral_R2'), spectral_R2 = outDPCA.spectral_R2; end
        if isfield(outDPCA, 'direct_Component_Corr'), direct_Component_Corr = outDPCA.direct_Component_Corr; end
        
   case 'ICA'
        [outICA] = runICAAnalysis(data.eeg_train, data.eeg_test, ...
             data.H_train, data.H_test, k, local_param, method_dir);
        out = outICA;
        if isfield(outICA, 'Comp_latent_matching_corr'), corr_table = outICA.Comp_latent_matching_corr; end
        if isfield(outICA, 'Comp_latent_matching_matrix'),   Comp_latent_matching_matrix   = outICA.Comp_latent_matching_matrix;   end
        if isfield(outICA, 'spectral_R2'), spectral_R2 = outICA.spectral_R2; end
        if isfield(outICA, 'direct_Component_Corr'), direct_Component_Corr = outICA.direct_Component_Corr; end
        
    case 'UMAP'
        n_neighbors = 3; min_dist = 0.9;
        [outUMAP] = runUMAPAnalysis( ...
            n_neighbors, min_dist, data.eeg_train, data.eeg_test, local_param, ...
            data.H_train, data.H_test, k, local_results_dir);
        out = outUMAP;
        if isfield(outUMAP, 'Comp_latent_matching_corr'), corr_table = outUMAP.Comp_latent_matching_corr; end
        if isfield(outUMAP, 'Comp_latent_matching_matrix'),    Comp_latent_matching_matrix   = outUMAP.Comp_latent_matching_matrix;    end
        if isfield(outUMAP, 'spectral_R2'), spectral_R2 = outUMAP.spectral_R2; end
        if isfield(outUMAP, 'direct_Component_Corr'), direct_Component_Corr = outUMAP.direct_Component_Corr; end
        
    case 'AE'
        [outAE] = runAutoencoderAnalysis(data.eeg_train, data.eeg_test,...
            data.H_train, data.H_test, k, local_param, local_results_dir);
        out = outAE;
        if isfield(outAE, 'Comp_latent_matching_corr'), corr_table = outAE.Comp_latent_matching_corr; end
        if isfield(outAE, 'Comp_latent_matching_matrix'),  Comp_latent_matching_matrix   = outAE.Comp_latent_matching_matrix;  end
        if isfield(outAE, 'spectral_R2'), spectral_R2 = outAE.spectral_R2; end
        if isfield(outAE, 'direct_Component_Corr'), direct_Component_Corr = outAE.direct_Component_Corr; end
        
    case 'iVAE'
        beta_val = 0.1; 
        [outIVAE] = runIVaeAnalysis(data.eeg_train, data.c_train, ...
                                             data.eeg_test, data.c_test, ...
                                             data.H_train, data.H_test, ...
                                             k, local_param, local_results_dir, beta_val);
        out = outIVAE;
        if isfield(outIVAE, 'Comp_latent_matching_corr'),   corr_table  = outIVAE.Comp_latent_matching_corr;   end
        if isfield(outIVAE, 'Comp_latent_matching_matrix'),      Comp_latent_matching_matrix    = outIVAE.Comp_latent_matching_matrix;      end
        if isfield(outIVAE, 'spectral_R2'), spectral_R2 = outIVAE.spectral_R2; end
        if isfield(outIVAE, 'direct_Component_Corr'), direct_Component_Corr = outIVAE.direct_Component_Corr; end
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
entry.Comp_latent_matching_corr = corr_table;
entry.Comp_latent_matching_matrix = Comp_latent_matching_matrix;
entry.direct_Component_Corr = direct_Component_Corr;
entry.spectral_R2 = spectral_R2;
entry.out  = out;
end