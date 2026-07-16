function [net, info] = trainEEG_VAE(X_train_1D, X_test_1D, cfg)
    arguments
        X_train_1D double 
        X_test_1D  double 
        cfg.bottleneckSize (1,1) double
        cfg.epochs double = 250
        
        % --- OPTIMIZED HYPERPARAMETERS ---
        cfg.batchSize double = 16
        cfg.learnRate double = 5e-4
        cfg.lambdaPSD double = 1.0             
        cfg.lambdaPower double = 2.0           
        cfg.beta double = 0.5            
        cfg.checkpointPath string = ""
    end
    
    WindowLength = size(X_train_1D, 2);
    TotalChannels = size(X_train_1D, 3); 
    
    lgraph = layerGraph();
    
    % ==========================================
    % 1. UNIFIED BROADBAND INPUT
    % ==========================================
    lgraph = addLayers(lgraph, imageInputLayer([1 WindowLength TotalChannels], "Normalization", "zscore", "Name", "input"));
    
    % ==========================================
    % 2. DEEP TEMPORAL ENCODER (Funneling DOWN)
    % Added Dropout and switched to Tanh to mirror successful CAE
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 15], 64, "Padding", "same", "Name", "enc_conv1") 
        groupNormalizationLayer("all-channels", "Name", "enc_gn1")
        tanhLayer("Name", "enc_tanh1")
        dropoutLayer(0.2, "Name", "dropout_enc1")
        
        convolution2dLayer([1 9], 32, "Padding", "same", "Name", "enc_conv2") 
        groupNormalizationLayer("all-channels", "Name", "enc_gn2")
        tanhLayer("Name", "enc_tanh2")
        dropoutLayer(0.2, "Name", "dropout_enc2")
        
        convolution2dLayer([1 5], 16, "Padding", "same", "Name", "enc_conv3")  
        groupNormalizationLayer("all-channels", "Name", "enc_gn3")
        tanhLayer("Name", "enc_tanh3")
        dropoutLayer(0.2, "Name", "dropout_enc3")
        
        % VAE BOTTLENECK
        convolution2dLayer([1 5], cfg.bottleneckSize * 2, "Padding", "same", "Name", "vae_parameters")
        vaeSamplingLayer("vae_sampler", cfg.bottleneckSize)
    ]);
    
    % ==========================================
    % 3. DEEP TEMPORAL DECODER (Funneling UP)
    % ==========================================
    lgraph = addLayers(lgraph, [
        convolution2dLayer([1 5], 16, "Padding", "same", "Name", "dec_conv1")  
        groupNormalizationLayer("all-channels", "Name", "dec_gn1")
        tanhLayer("Name", "dec_tanh1")
        
        convolution2dLayer([1 9], 32, "Padding", "same", "Name", "dec_conv2") 
        groupNormalizationLayer("all-channels", "Name", "dec_gn2")
        tanhLayer("Name", "dec_tanh2")
        
        convolution2dLayer([1 15], 64, "Padding", "same", "Name", "dec_conv3") 
        groupNormalizationLayer("all-channels", "Name", "dec_gn3")
        tanhLayer("Name", "dec_tanh3")
        
        % Final reconstruction back to raw EEG channels
        convolution2dLayer([1 1], TotalChannels, "Padding", "same", "Name", "reconstruction")
    ]);

    % ==========================================
    % 4. VAE LOSS HACK (Concatenation)
    % ==========================================
    lgraph = addLayers(lgraph, depthConcatenationLayer(2, "Name", "output_concat"));
    lgraph = addLayers(lgraph, vaeBroadbandLossLayer("vae_elbo_loss", TotalChannels, cfg.lambdaPSD, cfg.lambdaPower, cfg.beta, cfg.bottleneckSize));

    % ==========================================
    % 5. CONNECT THE GRAPH
    % ==========================================
    lgraph = connectLayers(lgraph, "input", "enc_conv1");
    lgraph = connectLayers(lgraph, "vae_sampler", "dec_conv1");
    lgraph = connectLayers(lgraph, "reconstruction", "output_concat/in1");
    lgraph = connectLayers(lgraph, "vae_parameters", "output_concat/in2"); 
    lgraph = connectLayers(lgraph, "output_concat", "vae_elbo_loss");
    
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
        "LearnRateSchedule", "piecewise", ...
        "LearnRateDropPeriod", 25, ...
        "LearnRateDropFactor", 0.5, ...
        "L2Regularization", 1e-4, ...        
        "Shuffle", "every-epoch", ...
        "Plots", "training-progress", ...
        "Verbose", false, ...
        "ValidationData", {X_test_1D, T_test_padded}, ...   
        "ValidationFrequency", val_freq, ...
        "ValidationPatience", 30, ...
        "CheckpointPath", cfg.checkpointPath, ...
        "OutputNetwork", "best-validation-loss", ...
        "ExecutionEnvironment", "auto");
    
    [net, info] = trainNetwork(X_train_1D, T_train_padded, lgraph, options); 
end