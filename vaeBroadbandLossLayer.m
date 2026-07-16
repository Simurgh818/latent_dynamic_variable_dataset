classdef vaeBroadbandLossLayer < nnet.layer.RegressionLayer
    properties
        TotalChannels
        LambdaPSD
        LambdaPower
        Beta
        BottleneckSize
    end
    methods
        function layer = vaeBroadbandLossLayer(name, totalChannels, lambdaPSD, lambdaPower, beta, bottleneckSize)
            layer.Name = name;
            layer.Description = "Broadband ELBO (Time + PSD + Power + KL Divergence)";
            layer.TotalChannels = totalChannels;
            layer.LambdaPSD = lambdaPSD;
            layer.LambdaPower = lambdaPower;
            layer.Beta = beta;
            layer.BottleneckSize = bottleneckSize;
        end
        
        function loss = forwardLoss(layer, Y, T)
            L = size(Y, 2); 
            BatchSize = size(Y, 4);
            k = layer.BottleneckSize;
            
            % 1. Unpack the Concatenation Hack
            Y_recon = Y(:,:, 1:layer.TotalChannels, :);
            Tb_all  = T(:,:, 1:layer.TotalChannels, :); 
            
            mu     = Y(:,:, layer.TotalChannels+1 : layer.TotalChannels+k, :);
            logVar = Y(:,:, layer.TotalChannels+k+1 : end, :);
            
            % 2. BROADBAND RECONSTRUCTION LOSS (Time + Freq + Power)
            % Time Loss
            time_loss = mean((Y_recon - Tb_all).^2, "all");
            
            % PSD Loss
            Y_fft = fft(Y_recon, [], 2) / L;
            T_fft = fft(Tb_all, [], 2) / L;
            Y_psd = real(Y_fft).^2 + imag(Y_fft).^2;
            T_psd = real(T_fft).^2 + imag(T_fft).^2;
            psd_loss = mean((Y_psd - T_psd).^2, "all");
            
            % Power Loss
            Y_power = mean(Y_recon.^2, 2);
            T_power = mean(Tb_all.^2, 2);
            power_loss = mean((Y_power - T_power).^2, "all");
            
            recon_loss = time_loss + (layer.LambdaPSD * psd_loss) + (layer.LambdaPower * power_loss);
            
            % 3. KL-DIVERGENCE LOSS
            kl_divergence = -0.5 * sum(1 + logVar - mu.^2 - exp(logVar), 'all');
            kl_loss_mean = kl_divergence / BatchSize;
            
            % 4. TOTAL ELBO LOSS
            loss = recon_loss + (layer.Beta * kl_loss_mean);
        end
        
        % FIREWALL: Prevents imaginary gradient crashes
        function dLdY = backwardLoss(layer, Y, T)
            Y_dl = dlarray(Y);
            [~, dLdY_dl] = dlfeval(@vaeBroadbandLossLayer.lossAndGrad, layer, Y_dl, T);
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