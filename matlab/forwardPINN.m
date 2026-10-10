function y = forwardPINN(params, x)
% FORWARDPINN Forward pass through the PINN network
%   y = forwardPINN(params, x)
%
% Architecture: Feedforward NN with Tanh activations + skip connection
%   y = NN(x) + 0.1 * Linear(x)
%
% Args:
%   params - Network parameters struct with fields:
%            fc1_W, fc1_b, ..., fcN_W, fcN_b (hidden layers)
%            fcout_W, fcout_b (output layer)
%            skip_W, skip_b (skip connection)
%            numHiddenLayers (integer)
%   x      - Input dlarray [batch_size x 4]
%
% Returns:
%   y - Output dlarray [batch_size x 5]: [cs, ce, phie, phis, JLi]

    numHL = 1;
    while isfield(params, sprintf('fc%d_W', numHL + 1))
        numHL = numHL + 1;
    end
    
    % Skip connection: identity = x * W' + b
    identity = x * params.skip_W.' + params.skip_b;
    
    % First hidden layer
    h = tanh(x * params.fc1_W.' + params.fc1_b);
    
    % Remaining hidden layers
    for i = 2:numHL
        W = params.(sprintf('fc%d_W', i));
        b = params.(sprintf('fc%d_b', i));
        h = tanh(h * W.' + b);
    end
    
    % Output layer (no activation)
    h = h * params.fcout_W.' + params.fcout_b;
    
    % Combine with skip connection
    y = h + 0.1 .* identity;
end
