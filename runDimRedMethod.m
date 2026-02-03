function entry = runDimRedMethod( ...
    method, data, local_param, k, ki, condition, dataset_name, ...
    method_dir, local_results_dir)

% Initialize generic temporary variables
R2  = NaN;
MSE = NaN;
out = struct();
corr_table = table(); % default empty
R_matrix = [];

% Run Analysis
switch method
    case 'PCA'
        [R2_test, MSE_test, outPCA] = runPCAAnalysis(data.eeg_train, data.eeg_test,...
            data.H_train, data.H_test, local_param, k, method_dir);
        
        % Extract Scalar Values immediately
        R2  = mean(R2_test(ki,:), 'omitnan');
        MSE = mean(MSE_test(ki,:), 'omitnan');

        out = outPCA;
        if isfield(outPCA, 'corr_PCA'), corr_table = outPCA.corr_PCA; end
        if isfield(outPCA, 'R_full'),   R_matrix   = outPCA.R_full;   end
        
    case 'AE'
        [R2, MSE, outAE] = runAutoencoderAnalysis(data.eeg_train, data.eeg_test,...
            data.H_train, data.H_test, k, local_param, local_results_dir);
        out = outAE;
        if isfield(outAE, 'corr_AE'), corr_table = outAE.corr_AE; end
        if isfield(outAE, 'R_full'),  R_matrix   = outAE.R_full;  end

    case 'ICA'
        [R2, MSE, outICA] = runICAAnalysis(data.eeg_train, data.eeg_test, ...
             data.H_train, data.H_test, k, local_param, method_dir);
        out = outICA;
        if isfield(outICA, 'corr_ICA'), corr_table = outICA.corr_ICA; end
        if isfield(outICA, 'R_full'),   R_matrix   = outICA.R_full;   end

    case 'UMAP'
        % Note: Java properties should ideally be set outside parfor, 
        % but some workers might need it reset.
        n_neighbors = 3; min_dist = 0.99;
        [R2, MSE, outUMAP] = runUMAPAnalysis( ...
            n_neighbors, min_dist, data.eeg_train, data.eeg_test, local_param, ...
            data.H_train, data.H_test, k, local_results_dir);
        out = outUMAP;
        if isfield(outUMAP, 'corr_UMAP'), corr_table = outUMAP.corr_UMAP; end
        if isfield(outUMAP, 'R_full'),    R_matrix   = outUMAP.R_full;    end

    case 'dPCA'
        [R2_test, MSE_test, outDPCA] = rundPCAAnalysis( ...
            data.eeg_train, data.H_train, local_param, k, method_dir);
        R2  = mean(R2_test(ki,:), 'omitnan');
        MSE = mean(MSE_test(ki,:), 'omitnan');

        out = outDPCA;
        if isfield(outDPCA, 'corr_dPCA'), corr_table = outDPCA.corr_dPCA; end
        if isfield(outDPCA, 'R_full'),    R_matrix   = outDPCA.R_full;    end
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
entry.out  = out;

end