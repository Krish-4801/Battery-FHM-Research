function metrics = computeValMetrics(params, inputs, cs_data, ce_data, phie_data, phis_data, JLi_data)
% COMPUTEVALMETRICS Evaluate model predictions against target data
%
% Args:
%   params      - Network parameters struct
%   inputs      - Input array [N x 4] (normalized space)
%   cs_data     - Target solid phase concentration (normalized)
%   ce_data     - Target electrolyte concentration (normalized)
%   phie_data   - Target electrolyte potential (normalized)
%   phis_data   - Target solid phase potential (normalized)
%   JLi_data    - Target intercalation current density (normalized)
%
% Returns:
%   metrics - Struct containing loss, mse, rmse, mae, r2 for each variable and totals

    % Forward pass (using dlarray)
    inputs_dl = dlarray(single(inputs));
    outputs_dl = forwardPINN(params, inputs_dl);
    outputs = double(extractdata(outputs_dl));

    % Separate predictions
    cs_pred   = outputs(:, 1);
    ce_pred   = outputs(:, 2);
    phie_pred = outputs(:, 3);
    phis_pred = outputs(:, 4);
    JLi_pred  = outputs(:, 5);

    % Convert targets to double
    targets = struct(...
        'cs', double(cs_data), ...
        'ce', double(ce_data), ...
        'phie', double(phie_data), ...
        'phis', double(phis_data), ...
        'JLi', double(JLi_data));

    preds = struct(...
        'cs', cs_pred, ...
        'ce', ce_pred, ...
        'phie', phie_pred, ...
        'phis', phis_pred, ...
        'JLi', JLi_pred);

    var_names = {'cs', 'ce', 'phie', 'phis', 'JLi'};

    metrics = struct();
    metrics.loss = struct();
    metrics.mse  = struct();
    metrics.rmse = struct();
    metrics.mae  = struct();
    metrics.r2   = struct();

    total_loss = 0;
    total_mse  = 0;
    total_mae  = 0;
    total_r2   = 0;

    for i = 1:length(var_names)
        name = var_names{i};
        y_true = targets.(name);
        y_pred = preds.(name);

        % L1 loss / MAE
        err_abs = abs(y_pred - y_true);
        mae_val = mean(err_abs);
        
        % MSE / RMSE
        err_sq = (y_pred - y_true).^2;
        mse_val = mean(err_sq);
        rmse_val = sqrt(mse_val);

        % R2
        ss_tot = sum((y_true - mean(y_true)).^2);
        ss_res = sum((y_true - y_pred).^2);
        r2_val = 1 - (ss_res / (ss_tot + 1e-8));

        metrics.loss.(name) = mae_val;
        metrics.mse.(name)  = mse_val;
        metrics.rmse.(name) = rmse_val;
        metrics.mae.(name)  = mae_val;
        metrics.r2.(name)   = r2_val;

        total_loss = total_loss + mae_val;
        total_mse  = total_mse + mse_val;
        total_mae  = total_mae + mae_val;
        total_r2   = total_r2 + r2_val;
    end

    num_vars = length(var_names);
    metrics.totalLoss = total_loss;
    metrics.totalMse  = total_mse / num_vars;
    metrics.totalRmse = sqrt(metrics.totalMse);
    metrics.totalMae  = total_mae / num_vars;
    metrics.totalR2   = total_r2 / num_vars;
    metrics.batchSize = size(inputs, 1);
end
