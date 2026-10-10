function params = initNetwork(inFeatures, hiddenFeatures, hiddenLayers, outFeatures)
% INITNETWORK Initializes PINN network parameters.
%   params = initNetwork(inFeatures, hiddenFeatures, hiddenLayers, outFeatures)
%   returns a struct containing the weights and biases for the neural network.
%   Weights are initialized with Xavier normal, biases with normal(0, 0.01).
%   All parameters are stored as single-precision dlarrays.

    params = struct();
    
    % Initialize input layer
    fan_in = inFeatures;
    fan_out = hiddenFeatures;
    std_val = sqrt(2 / (fan_in + fan_out));
    params.fc1_W = dlarray(single(std_val * randn(fan_out, fan_in)));
    params.fc1_b = dlarray(single(0.01 * randn(1, fan_out)));
    
    % Initialize hidden layers
    for i = 2:hiddenLayers
        fan_in = hiddenFeatures;
        fan_out = hiddenFeatures;
        std_val = sqrt(2 / (fan_in + fan_out));
        
        W_name = sprintf('fc%d_W', i);
        b_name = sprintf('fc%d_b', i);
        
        params.(W_name) = dlarray(single(std_val * randn(fan_out, fan_in)));
        params.(b_name) = dlarray(single(0.01 * randn(1, fan_out)));
    end
    
    % Initialize output layer
    fan_in = hiddenFeatures;
    fan_out = outFeatures;
    std_val = sqrt(2 / (fan_in + fan_out));
    params.fcout_W = dlarray(single(std_val * randn(fan_out, fan_in)));
    params.fcout_b = dlarray(single(0.01 * randn(1, fan_out)));
    
    % Initialize skip connection
    params.skip_W = dlarray(single(0.01 * randn(outFeatures, inFeatures)));
    params.skip_b = dlarray(single(zeros(1, outFeatures)));
end
