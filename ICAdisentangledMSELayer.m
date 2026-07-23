classdef ICAdisentangledMSELayer < nnet.layer.RegressionLayer
    properties
        LambdaICA
        BottleneckSize
        TotalChannels
    end
    
    methods
        function layer = ICAdisentangledMSELayer(name, lambdaICA, bottleneckSize, totalChannels)
            layer.Name = name;
            layer.Description = "Broadband MSE + ICA Cross-Covariance Penalty";
            layer.LambdaICA = lambdaICA;
            layer.BottleneckSize = bottleneckSize;
            layer.TotalChannels = totalChannels;
        end
        
        function loss = forwardLoss(layer, Y, T)
            k = layer.BottleneckSize;
            ch = layer.TotalChannels;
            
            % 1. Unpack the Concatenation Hack from Y
            % Y is [1 x Time x (Channels + k) x Batch]
            Y_recon = Y(:,:, 1:ch, :);      % Reconstructed EEG
            T_recon = T(:,:, 1:ch, :);      % True EEG (from Padded Target)
            
            Z = Y(:,:, ch+1 : end, :);      % The k bottleneck latents
            
            % ==========================================
            % 2. BROADBAND RECONSTRUCTION LOSS (Standard MSE)
            % ==========================================
            mse_loss = mean((Y_recon - T_recon).^2, "all");
            
            % ==========================================
            % 3. ICA CROSS-COVARIANCE PENALTY
            % ==========================================
            % Reshape Z from [1 x Time x k x Batch] to [k x (Time*Batch)]
            Z_flat = reshape(permute(Z, [3, 2, 4, 1]), [k, size(Z,2) * size(Z,4)]);
            
            % Center the latent variables
            M = size(Z_flat, 2);
            Z_centered = Z_flat - mean(Z_flat, 2);
            
            % Calculate Covariance Matrix [k x k]
            CovMatrix = (Z_centered * Z_centered') / (M - 1);
            
            % Mask out the diagonal elements (we only penalize off-diagonal covariance)
            % FIXED: Removed 'like' argument. MATLAB natively broadcasts standard arrays!
            mask = 1 - eye(k); 
            Cov_off_diagonal = CovMatrix .* mask;
            
            % ICA Loss: Mean squared off-diagonal covariance
            ica_loss = mean(Cov_off_diagonal.^2, 'all');
            
            % ==========================================
            % 4. TOTAL LOSS
            % ==========================================
            loss = mse_loss + (layer.LambdaICA * ica_loss);
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            Y_dl = dlarray(Y);
            [~, dLdY_dl] = dlfeval(@ICAdisentangledMSELayer.lossAndGrad, layer, Y_dl, T);
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