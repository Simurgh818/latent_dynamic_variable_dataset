function [net, info] = trainEEG_VAE_2(X_train_1D, X_test_1D, cfg)
    arguments
        X_train_1D double 
        X_test_1D  double 
        cfg.bottleneckSize (1,1) double
        cfg.epochs double = 250
        cfg.batchSize double = 32         
        cfg.learnRate double = 1e-3       
        cfg.beta double = 0.5            
        cfg.checkpointPath string = ""
    end
    
    WindowLength  = size(X_train_1D, 2);
    TotalChannels = size(X_train_1D, 3); 
    
    lgraph = layerGraph();
    
    % ==========================================
    % 1. UNIFIED BROADBAND INPUT
    % ==========================================
    lgraph = addLayers(lgraph, imageInputLayer([1 WindowLength TotalChannels], "Normalization", "zscore", "Name", "input"));
    
    % ==========================================
    % 2. 1D TEMPORAL ENCODER
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 15], 256, "Padding", "same", "Name", "enc_conv1") 
        leakyReluLayer(0.01, "Name", "enc_leakyrelu1")
        
        % VAE Bottleneck Parameters (Mean and Log-Variance) -> Outputs 2*k channels
        convolution2dLayer([1 1], cfg.bottleneckSize * 2, "Padding", "same", "Name", "vae_parameters")
        vaeSamplingLayer("vae_sampler", cfg.bottleneckSize)
    ]);
    
    % ==========================================
    % 3. 1D TEMPORAL DECODER
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 15], 256, "Padding", "same", "Name", "dec_conv1")  
        leakyReluLayer(0.01, "Name", "dec_leakyrelu1")
        
        % Final reconstruction back to raw EEG channels
        convolution2dLayer([1 1], TotalChannels, "Padding", "same", "Name", "reconstruction")
    ]);
    
    % ==========================================
    % 4. THE LOSS HACK (Concatenation & Single Loss Layer)
    % ==========================================
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, "Name", "output_concat"));
    lgraph = addLayers(lgraph, vaeMSEAndKLLossLayer("vae_loss", TotalChannels, cfg.beta, cfg.bottleneckSize)); % disentangledMSELayer
    
    % ==========================================
    % 5. CONNECT THE GRAPH
    % ==========================================
    lgraph = connectLayers(lgraph, "input", "enc_conv1");
    lgraph = connectLayers(lgraph, "vae_sampler", "dec_conv1");
    
    % Hook up the hack: combine reconstruction and vae_parameters into the loss
    lgraph = connectLayers(lgraph, "reconstruction", "output_concat/in1");
    lgraph = connectLayers(lgraph, "vae_parameters", "output_concat/in2"); 
    lgraph = connectLayers(lgraph, "output_concat", "vae_loss");
    
    % ==========================================
    % 6. PADDING TARGET DATA & TRAINING
    % ==========================================
    pad_train = zeros(1, WindowLength, cfg.bottleneckSize * 2, size(X_train_1D, 4));
    pad_test  = zeros(1, WindowLength, cfg.bottleneckSize * 2, size(X_test_1D, 4));
    
    T_train_padded = cat(3, X_train_1D, pad_train);
    T_test_padded  = cat(3, X_test_1D, pad_test);
    
    val_freq = max(1, floor(size(X_train_1D, 4) / cfg.batchSize));
    
    options = trainingOptions("adam", ...
        "MaxEpochs", cfg.epochs, ...
        "MiniBatchSize", cfg.batchSize, ...
        "InitialLearnRate", cfg.learnRate, ...
        "L2Regularization", 0, ...              
        "Shuffle", "every-epoch", ...
        "Plots", "training-progress", ... %"none"
        "Verbose", false, ...
        "ValidationData", {X_test_1D, T_test_padded}, ...   
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ...
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    [net, info] = trainNetwork(X_train_1D, T_train_padded, lgraph, options); 
end