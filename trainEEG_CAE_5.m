function [net, info] = trainEEG_CAE_5(X_train_1D, X_test_1D, cfg)
    arguments
        X_train_1D double 
        X_test_1D  double 
        cfg.bottleneckSize (1,1) double
        
        % --- HYPERPARAMETERS FOR CUSTOM LOSS ---
        cfg.numBands double = 5
        cfg.bandWeights double = [1 1 1 2 3]       
        cfg.lambdaICA double = 0.1    % Penalty weight for ICA cross-covariance        
        
        % --- TRAINING HYPERPARAMETERS ---
        cfg.epochs double = 250
        cfg.batchSize double = 32
        cfg.learnRate double = 1e-3
        cfg.checkpointPath string = ""
    end
    
    WindowLength = size(X_train_1D, 2);
    TotalChannels = size(X_train_1D, 3);
    NumChannelsPerBand = TotalChannels / cfg.numBands; 
    
    lgraph = layerGraph();
    
    % ==========================================
    % 1. UNIFIED BROADBAND INPUT
    % ==========================================
    lgraph = addLayers(lgraph, imageInputLayer([1 WindowLength TotalChannels], "Normalization", "zscore", "Name", "input"));
    
    % ==========================================
    % 2. 1D TEMPORAL ENCODER (Single Branch, Matched to CAE)
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 15], 256, "Padding", "same", "Name", "enc_conv1")
        leakyReluLayer(0.01, "Name", "enc_leakyrelu1")
        
        % BOTTLENECK: 1x1 kernel mapping 256 feature maps into k latents
        convolution2dLayer([1 1], cfg.bottleneckSize, "Padding", "same", "Name", "bottleneck")
    ]);

    % ==========================================
    % 3. 1D TEMPORAL DECODER (Single Branch, Matched to CAE)
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 15], 256, "Padding", "same", "Name", "dec_conv1")
        leakyReluLayer(0.01, "Name", "dec_leakyrelu1")
        
        % Final reconstruction back to raw stacked EEG channels (155 channels)
        convolution2dLayer([1 1], TotalChannels, "Padding", "same", "Name", "reconstruction")
    ]);

    % ==========================================
    % 4. ADDITION OF THE ICA OUTPUT HACK
    % ==========================================
    % --> THE HACK: Concatenate the reconstruction and the bottleneck together
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, "Name", "output_concat"));
    
    % The custom loss layer parses the top 155 channels for band-wise MSE 
    % and the bottom 'k' channels for the ICA penalty
    % Initialize the simplified MSE + ICA layer using the new filename
    lgraph = addLayers(lgraph, ICAdisentangledMSELayer("ica_mse_loss", cfg.lambdaICA, cfg.bottleneckSize, TotalChannels));
    
    % ==========================================
    % 5. CONNECT THE GRAPH 
    % ==========================================
    % Connect main sequential pipeline
    lgraph = connectLayers(lgraph, "input", "enc_conv1");
    lgraph = connectLayers(lgraph, "bottleneck", "dec_conv1");
    
    % Connect the Hack Layers
    lgraph = connectLayers(lgraph, "reconstruction", "output_concat/in1");
    lgraph = connectLayers(lgraph, "bottleneck", "output_concat/in2");
    lgraph = connectLayers(lgraph, "output_concat", "ica_mse_loss"); % <--- FIXED!
    
    % ==========================================
    % 6. PADDING TARGET DATA (To bypass MATLAB restrictions)
    % ==========================================
    % We must pad the target tensors with fake "k" channels so their size 
    % matches the output_concat size. (These zeros are ignored in the loss layer).
    pad_train = zeros(1, WindowLength, cfg.bottleneckSize, size(X_train_1D, 4));
    pad_test  = zeros(1, WindowLength, cfg.bottleneckSize, size(X_test_1D, 4));
    
    T_train_padded = cat(3, X_train_1D, pad_train);
    T_test_padded  = cat(3, X_test_1D, pad_test);
    
    % ==========================================
    % 7. TRAINING OPTIONS
    % ==========================================
    val_freq = max(1, floor(size(X_train_1D, 4) / cfg.batchSize));
    
    options = trainingOptions("adam", ...
        "MaxEpochs", cfg.epochs, ...
        "MiniBatchSize", cfg.batchSize, ...
        "InitialLearnRate", cfg.learnRate, ...
        "LearnRateSchedule", "piecewise", ...
        "LearnRateDropPeriod", 25, ...
        "LearnRateDropFactor", 0.5, ...
        "L2Regularization", 0, ...              % Changed to 0 (Matched to CAE)
        "Shuffle", "every-epoch", ...
        "Plots", "training-progress", ... % "none"
        "Verbose", false, ...
        "ValidationData", {X_test_1D, T_test_padded}, ...   % <-- Using Padded Target
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ...
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    [net, info] = trainNetwork(X_train_1D, T_train_padded, lgraph, options);
end