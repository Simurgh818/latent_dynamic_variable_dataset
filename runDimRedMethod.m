function entry = runDimRedMethod( ...
    method, data, local_param, k, ki, condition, dataset_name, ...
    method_dir, local_results_dir)

% Initialize generic temporary variables
R2  = NaN;
MSE = NaN;
out = struct();
corr_table = table(); % default empty
R_matrix = [];
spectral_R2 = NaN; % Initialize spectral_R2 for cases where it's not defined

% Run Analysis
switch method
    case 'PCA'
        [R2, MSE, outPCA] = runPCAAnalysis(data.eeg_train, data.eeg_test,...
            data.H_train, data.H_test, local_param, k, method_dir);
                        
        out = outPCA;
        if isfield(outPCA, 'corr_PCA'), corr_table = outPCA.corr_PCA; end
        if isfield(outPCA, 'R_full'),   R_matrix   = outPCA.R_full;   end
        if isfield(outPCA, 'spectral_R2'), spectral_R2 = outPCA.spectral_R2; end
        if isfield(outPCA, 'zeroLagCorr'), zeroLagCorr = outPCA.zeroLagCorr; end

   case 'dPCA'
        % Assuming rundPCAAnalysis also takes 'k' now and returns single-k result
        [R2, MSE, outDPCA] = rundPCAAnalysis( ...
            data.eeg_train, data.eeg_test, data.H_train, data.H_test,...
            local_param, k, method_dir);       
        
        out = outDPCA;
        if isfield(outDPCA, 'corr_dPCA'), corr_table = outDPCA.corr_dPCA; end
        if isfield(outDPCA, 'R_full'),    R_matrix   = outDPCA.R_full;    end
        if isfield(outDPCA, 'spectral_R2'), spectral_R2 = outDPCA.spectral_R2; end
        if isfield(outDPCA, 'zeroLagCorr'), zeroLagCorr = outDPCA.zeroLagCorr; end

   case 'ICA'
        [R2, MSE, outICA] = runICAAnalysis(data.eeg_train, data.eeg_test, ...
             data.H_train, data.H_test, k, local_param, method_dir);
        out = outICA;
        if isfield(outICA, 'corr_ICA'), corr_table = outICA.corr_ICA; end
        if isfield(outICA, 'R_full'),   R_matrix   = outICA.R_full;   end
        if isfield(outICA, 'spectral_R2'), spectral_R2 = outICA.spectral_R2; end
        if isfield(outICA, 'zeroLagCorr'), zeroLagCorr = outICA.zeroLagCorr; end
    
    case 'AE'
        [R2, MSE, outAE] = runAutoencoderAnalysis(data.eeg_train, data.eeg_test,...
            data.H_train, data.H_test, k, local_param, local_results_dir);
        out = outAE;
        if isfield(outAE, 'corr_AE'), corr_table = outAE.corr_AE; end
        if isfield(outAE, 'R_full'),  R_matrix   = outAE.R_full;  end
        if isfield(outAE, 'spectral_R2'), spectral_R2 = outAE.spectral_R2; end
        if isfield(outAE, 'zeroLagCorr'), zeroLagCorr = outAE.zeroLagCorr; end
        
    case 'UMAP'
        n_neighbors = 3; min_dist = 0.99;
        [R2, MSE, outUMAP] = runUMAPAnalysis( ...
            n_neighbors, min_dist, data.eeg_train, data.eeg_test, local_param, ...
            data.H_train, data.H_test, k, local_results_dir);
        out = outUMAP;
        if isfield(outUMAP, 'corr_UMAP'), corr_table = outUMAP.corr_UMAP; end
        if isfield(outUMAP, 'R_full'),    R_matrix   = outUMAP.R_full;    end
        if isfield(outUMAP, 'spectral_R2'), spectral_R2 = outUMAP.spectral_R2; end
        if isfield(outUMAP, 'zeroLagCorr'), zeroLagCorr = outUMAP.zeroLagCorr; end
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
entry.stats.R2  = R2;
entry.stats.MSE = MSE;
entry.corr = corr_table;
entry.R_matrix = R_matrix;
entry.zeroLagCorr = zeroLagCorr;
entry.spectral_R2 = spectral_R2;
entry.out  = out;

end