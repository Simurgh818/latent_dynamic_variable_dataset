function [net, info] = trainEEG_CAE_5(X_train_1D, X_test_1D, cfg)
    arguments
        X_train_1D single 
        X_test_1D  single 
        cfg.bottleneckSize (1,1) double
        
        % --- HYPERPARAMETERS FOR CUSTOM LOSS ---
        cfg.lambdaICA double = 0.1    % Penalty weight for ICA cross-covariance        
        
        % --- TRAINING HYPERPARAMETERS ---
        cfg.epochs double = 250
        cfg.batchSize double = 32
        cfg.learnRate double = 1e-3
        cfg.checkpointPath string = ""
        cfg.maxValWindows double = 256 % Cap validation windows during training to prevent OOM
    end
    
    WindowLength  = size(X_train_1D, 2);
    TotalChannels = size(X_train_1D, 3);
    numTrainWins  = size(X_train_1D, 4);
    numTestWins   = size(X_test_1D, 4);
    
    lgraph = layerGraph();
    
    % ==========================================
    % 1. UNIFIED BROADBAND INPUT
    % ==========================================
    lgraph = addLayers(lgraph, imageInputLayer([1 WindowLength TotalChannels], "Normalization", "zscore", "Name", "input"));
    
    % ==========================================
    % 2. 1D TEMPORAL ENCODER (Single Branch, Matched to CAE)
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 5], 512, "Padding", "same", "Name", "enc_conv1")
        leakyReluLayer(0.01, "Name", "enc_leakyrelu1")
        convolution2dLayer([1 1], cfg.bottleneckSize, "Padding", "same", "Name", "bottleneck")
    ]);
    
    % ==========================================
    % 3. 1D TEMPORAL DECODER (Single Branch, Matched to CAE)
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 5], 512, "Padding", "same", "Name", "dec_conv1")
        leakyReluLayer(0.01, "Name", "dec_leakyrelu1")
        convolution2dLayer([1 1], TotalChannels, "Padding", "same", "Name", "reconstruction")
    ]);
    
    % ==========================================
    % 4. ADDITION OF THE ICA OUTPUT HACK
    % ==========================================
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, "Name", "output_concat"));
    lgraph = addLayers(lgraph, ICAdisentangledMSELayer("ica_mse_loss", cfg.lambdaICA, cfg.bottleneckSize, TotalChannels));
    
    % ==========================================
    % 5. CONNECT THE GRAPH 
    % ==========================================
    lgraph = connectLayers(lgraph, "input", "enc_conv1");
    lgraph = connectLayers(lgraph, "bottleneck", "dec_conv1");
    lgraph = connectLayers(lgraph, "reconstruction", "output_concat/in1");
    lgraph = connectLayers(lgraph, "bottleneck", "output_concat/in2");
    lgraph = connectLayers(lgraph, "output_concat", "ica_mse_loss");
    
    % ==========================================
    % 6. PADDING TARGET DATA (In Single Precision & Subsampled Validation)
    % ==========================================
    % Subsample X_test_1D for early-stopping validation to prevent forwardLoss OOM
    if numTestWins > cfg.maxValWindows
        rng(42); % Fixed seed for reproducible validation subset across runs
        val_idx = randperm(numTestWins, cfg.maxValWindows);
        X_val_1D = X_test_1D(:, :, :, val_idx);
    else
        X_val_1D = X_test_1D;
    end
    clear X_test_1D; % Free full test copy inside local workspace
    
    % Allocate padded targets directly in single precision without intermediate pad arrays
    T_train_padded = cat(3, X_train_1D, zeros(1, WindowLength, cfg.bottleneckSize, numTrainWins, 'single'));
    T_val_padded   = cat(3, X_val_1D,   zeros(1, WindowLength, cfg.bottleneckSize, size(X_val_1D, 4), 'single'));
    
    % ==========================================
    % 7. TRAINING OPTIONS
    % ==========================================
    val_freq = max(1, floor(numTrainWins / cfg.batchSize));
    
    % Prevent multiple parfor workers from fighting over a single GPU's memory
    if ~isempty(getCurrentTask())
        exec_env = "cpu";
    else
        exec_env = "auto";
    end
    
    options = trainingOptions("adam", ...
        "MaxEpochs", cfg.epochs, ...
        "MiniBatchSize", cfg.batchSize, ...
        "InitialLearnRate", cfg.learnRate, ...
        "LearnRateSchedule", "piecewise", ...
        "LearnRateDropPeriod", 25, ...
        "LearnRateDropFactor", 0.5, ...
        "L2Regularization", 0, ...
        "Shuffle", "every-epoch", ...
        "Plots", "none", ...
        "Verbose", false, ...
        "ValidationData", {X_val_1D, T_val_padded}, ...
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ...
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", exec_env);
    
    [net, info] = trainNetwork(X_train_1D, T_train_padded, lgraph, options);
end