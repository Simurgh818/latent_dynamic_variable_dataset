classdef vaeMSEAndKLLossLayer < nnet.layer.RegressionLayer
    properties
        TotalChannels
        Beta
        BottleneckSize
    end
    methods
        function layer = vaeMSEAndKLLossLayer(name, totalChannels, beta, bottleneckSize)
            layer.Name = name;
            layer.Description = "Combined Standard MSE + KL Divergence Loss";
            layer.TotalChannels = totalChannels;
            layer.Beta = beta;
            layer.BottleneckSize = bottleneckSize;
        end
        
        function loss = forwardLoss(layer, Y, T)
            BatchSize = size(Y, 4);
            k = layer.BottleneckSize;
            ch = layer.TotalChannels;
            
            % 1. Unpack the Concatenation Hack from Y (Predictions) and T (Targets)
            Y_recon = Y(:,:, 1:ch, :);
            T_recon = T(:,:, 1:ch, :); 
            
            mu_pred     = Y(:,:, ch+1 : ch+k, :);
            logVar_pred = Y(:,:, ch+k+1 : end, :);
            
            % 2. RECONSTRUCTION LOSS (Standard Mean Squared Error)
            mse_loss = mean((Y_recon - T_recon).^2, "all");
            
            % 3. KL-DIVERGENCE LOSS
            kl_divergence = -0.5 * sum(1 + logVar_pred - mu_pred.^2 - exp(logVar_pred), 'all');
            kl_loss_mean = kl_divergence / BatchSize;
            
            % 4. TOTAL VAE LOSS
            loss = mse_loss + (layer.Beta * kl_loss_mean);
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            Y_dl = dlarray(Y);
            [~, dLdY_dl] = dlfeval(@vaeMSEAndKLLossLayer.lossAndGrad, layer, Y_dl, T);
            dLdY = real(extractdata(dLdY_dl));
        end
    end
    
    methods (Static)
        function [loss, dLdY] = lossAndGrad(layer, Y_dl, T)
            loss = layer.forwardLoss(Y_dl, T);
            dLdY = dlgradient(loss, Y_dl);
        end
    end
end