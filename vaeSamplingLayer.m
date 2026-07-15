classdef vaeSamplingLayer < nnet.layer.Layer
    properties
        K % Bottleneck size
    end
    methods
        function layer = vaeSamplingLayer(name, k)
            layer.Name = name;
            layer.K = k;
            layer.Description = "Zhao 2024 Reparameterization Trick";
        end
        
        function Z = predict(layer, X)
            % X contains 2*k channels: the first half is Mu, the second half is log(Sigma^2)
            mu = X(:,:, 1:layer.K, :);
            logVar = X(:,:, layer.K+1:end, :);
            
            % Generate Gaussian noise (epsilon)
            epsilon = randn(size(mu), 'like', X);
            
            % Z = mu + sigma * epsilon
            Z = mu + exp(0.5 * logVar) .* epsilon;
        end
    end
end