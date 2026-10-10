%% PINN Battery FHM Solver - Main Script
% =========================================================================
% Replication of the Python/PyTorch PINN-based Full Homogenized Macroscale
% (FHM) model solver for lithium-ion batteries.
%
% Reference: "Real-Time Discharge Curve and State of Charge Estimation of
% Lithium-Ion Batteries via a Physics-Informed Full Homogenized Macroscale
% Model" - Journal of Energy Storage, Abbasi et al.
%
% Requirements: MATLAB R2023b+, Deep Learning Toolbox
% =========================================================================

clear; clc; close all;

%% ========================================================================
%  STEP 1: Data Preparation
% =========================================================================
fprintf('=== Step 1: Loading and Preprocessing Data ===\n');

% Battery parameters
bp = batteryParams();

% Script directory
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

% Load data (requires Data.mat - see convertPklToMat.py)
dataPath = fullfile(scriptDir, '..', 'Data.mat');
if ~isfile(dataPath)
    dataPath = fullfile(pwd, '..', 'Data.mat');
end
if ~isfile(dataPath)
    dataPath = fullfile(pwd, 'Data.mat');
end
[trainData, valData, testData, ns] = loadData(dataPath);

fprintf('Training samples:   %d\n', size(trainData.inputs, 1));
fprintf('Validation samples: %d\n', size(valData.inputs, 1));
fprintf('Test samples:       %d\n', size(testData.inputs, 1));

%% ========================================================================
%  STEP 2: Network Initialization
% =========================================================================
fprintf('\n=== Step 2: Initializing PINN Network ===\n');

inFeatures     = 4;   % [x, t, C, T]
hiddenFeatures = 256;
hiddenLayers   = 6;
outFeatures    = 5;   % [cs, ce, phie, phis, JLi]

params = initNetwork(inFeatures, hiddenFeatures, hiddenLayers, outFeatures);
fprintf('Network: %d -> %dx%d -> %d (with skip connection)\n', ...
    inFeatures, hiddenFeatures, hiddenLayers, outFeatures);

% Check for GPU
hasGPU = false;
try
    hasGPU = (gpuDeviceCount > 0);
catch
    hasGPU = false;
end

if hasGPU
    fprintf('GPU detected - using GPU acceleration\n');
    executionEnvironment = 'gpu';
    params = dlupdate(@gpuArray, params);
else
    fprintf('No GPU detected / CPU fallback - using CPU\n');
    executionEnvironment = 'cpu';
end

%% ========================================================================
%  STEP 3: Pretraining (Data-Only)
% =========================================================================
fprintf('\n=== Step 3: Pretraining Phase (Data-Only) ===\n');

% Pretraining hyperparameters
preConfig = struct();
preConfig.lr         = 5e-3;
preConfig.epochs     = 50;
preConfig.batchSize  = 8192;
preConfig.gradClip   = 1.0;

% Pretraining data loss weights
pw = struct('w1', 1.0, 'w2', 1.0, 'w3', 1.0, 'w4', 1.0, 'w5', 1.0);

% Initialize Adam state
avgGrad   = [];
avgSqGrad = [];
iteration = 0;

% Training history
preHistory = struct();
preHistory.trainLoss = zeros(preConfig.epochs, 1);
preHistory.valLoss   = zeros(preConfig.epochs, 1);
preHistory.lr        = zeros(preConfig.epochs, 1);
preHistory.r2        = zeros(preConfig.epochs, 5); % [cs, ce, phie, phis, JLi]

nTrain = size(trainData.inputs, 1);

for epoch = 1:preConfig.epochs
    epochLoss = 0;
    numBatches = 0;
    
    % Shuffle training data
    shuffleIdx = randperm(nTrain);
    
    for batchStart = 1:preConfig.batchSize:nTrain
        batchEnd = min(batchStart + preConfig.batchSize - 1, nTrain);
        idx = shuffleIdx(batchStart:batchEnd);
        batchSize = length(idx);
        
        % Extract batch (unformatted dlarrays)
        x_dl = dlarray(trainData.inputs(idx, 1));
        t_dl = dlarray(trainData.inputs(idx, 2));
        C_dl = dlarray(trainData.inputs(idx, 3));
        T_dl = dlarray(trainData.inputs(idx, 4));
        cs_d = dlarray(trainData.cs(idx));
        ce_d = dlarray(trainData.ce(idx));
        phie_d = dlarray(trainData.phie(idx));
        phis_d = dlarray(trainData.phis(idx));
        JLi_d  = dlarray(trainData.JLi(idx));
        
        if strcmp(executionEnvironment, 'gpu')
            x_dl = gpuArray(x_dl); t_dl = gpuArray(t_dl);
            C_dl = gpuArray(C_dl); T_dl = gpuArray(T_dl);
            cs_d = gpuArray(cs_d); ce_d = gpuArray(ce_d);
            phie_d = gpuArray(phie_d); phis_d = gpuArray(phis_d);
            JLi_d = gpuArray(JLi_d);
        end
        
        % Compute gradients
        [loss, grads] = dlfeval(@pretrainGradients, params, ...
            x_dl, t_dl, C_dl, T_dl, cs_d, ce_d, phie_d, phis_d, JLi_d, pw);
        
        % Gradient clipping
        grads = dlupdate(@(g) thresholdL2Norm(g, preConfig.gradClip), grads);
        
        % Adam update
        iteration = iteration + 1;
        [params, avgGrad, avgSqGrad] = adamupdate(params, grads, ...
            avgGrad, avgSqGrad, iteration, preConfig.lr);
        
        epochLoss = epochLoss + extractdata(loss) * batchSize;
        numBatches = numBatches + 1;
    end
    
    preHistory.trainLoss(epoch) = epochLoss / nTrain;
    preHistory.lr(epoch) = preConfig.lr;
    
    % Validation
    valMetrics = computeValMetrics(params, valData.inputs, ...
        valData.cs, valData.ce, valData.phie, valData.phis, valData.JLi);
    preHistory.valLoss(epoch) = valMetrics.totalLoss;
    preHistory.r2(epoch, :) = [valMetrics.r2.cs, valMetrics.r2.ce, ...
        valMetrics.r2.phie, valMetrics.r2.phis, valMetrics.r2.JLi];
    
    fprintf('Epoch %03d | Train: %.3e | Val: %.3e | R²: cs=%.4f ce=%.4f phie=%.4f phis=%.4f JLi=%.4f\n', ...
        epoch, preHistory.trainLoss(epoch), preHistory.valLoss(epoch), ...
        preHistory.r2(epoch, 1), preHistory.r2(epoch, 2), ...
        preHistory.r2(epoch, 3), preHistory.r2(epoch, 4), preHistory.r2(epoch, 5));
end

% Save pretrained model
savePath = scriptDir;
save(fullfile(savePath, 'pretrained_model.mat'), 'params');
fprintf('Pretrained model saved.\n');

%% ========================================================================
%  STEP 4: Main Training (Physics-Informed)
% =========================================================================
fprintf('\n=== Step 4: Main Training Phase (Physics-Informed) ===\n');

% Main training hyperparameters
config = struct();
config.lr          = 5e-4;
config.epochs      = 100;
config.batchSize   = 8192;
config.patience    = 100;
config.gradClip    = 1.0;
config.weightDecay = 1e-4;

% Physics loss weights (from grid search in the paper)
w = struct();
w.w1 = 8e-15;  % Residual loss
w.w2 = 1e-14;  % Boundary condition loss
w.w3 = 1e-2;   % Initial condition loss
w.w4 = 1.0;    % Data loss: cs
w.w5 = 2.0;    % Data loss: ce
w.w6 = 1.0;    % Data loss: phie
w.w7 = 1.0;    % Data loss: phis
w.w8 = 1.0;    % Data loss: JLi

% Reload pretrained model
loaded = load(fullfile(savePath, 'pretrained_model.mat'), 'params');
params = loaded.params;
if strcmp(executionEnvironment, 'gpu')
    params = dlupdate(@gpuArray, params);
end

% Reset Adam state for main training
avgGrad   = [];
avgSqGrad = [];
iteration = 0;

% Training history
history = struct();
history.trainLoss = zeros(config.epochs, 1);
history.valLoss   = zeros(config.epochs, 1);
history.lr        = zeros(config.epochs, 1);
history.r2        = zeros(config.epochs, 5);

bestValLoss = Inf;
epochsNoImprove = 0;

% OneCycle learning rate schedule
totalSteps = config.epochs * ceil(nTrain / config.batchSize);
lrSchedule = oneCycleLR(config.lr, totalSteps, 0.1, 25, 1000);
globalStep = 0;

for epoch = 1:config.epochs
    epochLoss = 0;
    
    % Shuffle
    shuffleIdx = randperm(nTrain);
    
    for batchStart = 1:config.batchSize:nTrain
        batchEnd = min(batchStart + config.batchSize - 1, nTrain);
        idx = shuffleIdx(batchStart:batchEnd);
        batchSize = length(idx);
        
        % Current learning rate from schedule
        globalStep = globalStep + 1;
        currentLR = lrSchedule(min(globalStep, totalSteps));
        
        % Extract batch (unformatted dlarrays)
        x_dl = dlarray(trainData.inputs(idx, 1));
        t_dl = dlarray(trainData.inputs(idx, 2));
        C_dl = dlarray(trainData.inputs(idx, 3));
        T_dl = dlarray(trainData.inputs(idx, 4));
        cs_d = dlarray(trainData.cs(idx));
        ce_d = dlarray(trainData.ce(idx));
        phie_d = dlarray(trainData.phie(idx));
        phis_d = dlarray(trainData.phis(idx));
        JLi_d  = dlarray(trainData.JLi(idx));
        
        if strcmp(executionEnvironment, 'gpu')
            x_dl = gpuArray(x_dl); t_dl = gpuArray(t_dl);
            C_dl = gpuArray(C_dl); T_dl = gpuArray(T_dl);
            cs_d = gpuArray(cs_d); ce_d = gpuArray(ce_d);
            phie_d = gpuArray(phie_d); phis_d = gpuArray(phis_d);
            JLi_d = gpuArray(JLi_d);
        end
        
        % Compute gradients (physics-informed)
        [loss, grads] = dlfeval(@physicsGradients, params, ...
            x_dl, t_dl, C_dl, T_dl, ...
            cs_d, ce_d, phie_d, phis_d, JLi_d, ns, bp, w);
        
        % Gradient clipping
        grads = dlupdate(@(g) thresholdL2Norm(g, config.gradClip), grads);
        
        % AdamW update (Adam + weight decay)
        iteration = iteration + 1;
        [params, avgGrad, avgSqGrad] = adamupdate(params, grads, ...
            avgGrad, avgSqGrad, iteration, currentLR);
        
        % Apply weight decay separately (AdamW style)
        params = dlupdate(@(p) p * (1 - currentLR * config.weightDecay), params);
        
        epochLoss = epochLoss + extractdata(loss) * batchSize;
    end
    
    history.trainLoss(epoch) = epochLoss / nTrain;
    history.lr(epoch) = currentLR;
    
    % Validation
    valMetrics = computeValMetrics(params, valData.inputs, ...
        valData.cs, valData.ce, valData.phie, valData.phis, valData.JLi);
    history.valLoss(epoch) = valMetrics.totalLoss;
    history.r2(epoch, :) = [valMetrics.r2.cs, valMetrics.r2.ce, ...
        valMetrics.r2.phie, valMetrics.r2.phis, valMetrics.r2.JLi];
    
    % Save best model
    if valMetrics.totalLoss < bestValLoss
        bestValLoss = valMetrics.totalLoss;
        save(fullfile(savePath, 'best_model.mat'), 'params');
        epochsNoImprove = 0;
    else
        epochsNoImprove = epochsNoImprove + 1;
    end
    
    fprintf('Epoch %03d | Train: %.3e | Val: %.3e | LR: %.2e | R²: cs=%.4f ce=%.4f phie=%.4f phis=%.4f JLi=%.4f\n', ...
        epoch, history.trainLoss(epoch), history.valLoss(epoch), currentLR, ...
        history.r2(epoch, 1), history.r2(epoch, 2), ...
        history.r2(epoch, 3), history.r2(epoch, 4), history.r2(epoch, 5));
    
    % Early stopping
    if epochsNoImprove >= config.patience
        fprintf('Early stopping at epoch %d\n', epoch);
        history.trainLoss = history.trainLoss(1:epoch);
        history.valLoss   = history.valLoss(1:epoch);
        history.lr        = history.lr(1:epoch);
        history.r2        = history.r2(1:epoch, :);
        break;
    end
end

%% ========================================================================
%  STEP 5: Test Evaluation
% =========================================================================
fprintf('\n=== Step 5: Test Evaluation ===\n');

% Load best model
loaded = load(fullfile(savePath, 'best_model.mat'), 'params');
params = loaded.params;

% Test metrics
testMetrics = computeValMetrics(params, testData.inputs, ...
    testData.cs, testData.ce, testData.phie, testData.phis, testData.JLi);

fprintf('\n===== TEST RESULTS =====\n');
fprintf('Total Loss: %.4f\n', testMetrics.totalLoss);
fprintf('\n%-8s %-10s %-10s %-10s\n', 'Variable', 'RMSE', 'R²', 'MAE');
fprintf('%s\n', repmat('-', 1, 40));
vars = {'cs', 'ce', 'phie', 'phis', 'JLi'};
for i = 1:5
    v = vars{i};
    fprintf('%-8s %-10.4f %-10.4f %-10.4f\n', v, ...
        testMetrics.rmse.(v), testMetrics.r2.(v), testMetrics.mae.(v));
end

%% ========================================================================
%  STEP 6: Training Curves
% =========================================================================
fprintf('\n=== Step 6: Plotting Training Curves ===\n');

plotTrainingCurves(history);

%% ========================================================================
%  STEP 7: Evaluate Discharge and SOC Curves
% =========================================================================
fprintf('\n=== Step 7: Generating Discharge and SOC Curves ===\n');

% Evaluate at T=298K, C=1 (default case from paper)
evaluateResults(params, ns, bp, dataPath, 298.0, 1);

fprintf('\n=== All Done! ===\n');

%% ========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function g = thresholdL2Norm(g, maxNorm)
% Clip gradient by L2 norm
    normG = sqrt(sum(g(:).^2));
    if normG > maxNorm
        g = g * (maxNorm / normG);
    end
end

function schedule = oneCycleLR(maxLR, totalSteps, pctStart, divFactor, finalDivFactor)
% OneCycle learning rate schedule matching PyTorch's OneCycleLR
% Returns a function handle: lr = schedule(step)
    warmupSteps = round(pctStart * totalSteps);
    initLR = maxLR / divFactor;
    finalLR = maxLR / (divFactor * finalDivFactor);
    
    schedule = @(step) computeLR(step, warmupSteps, totalSteps, initLR, maxLR, finalLR);
end

function lr = computeLR(step, warmupSteps, totalSteps, initLR, maxLR, finalLR)
    if step <= warmupSteps
        % Linear warmup
        lr = initLR + (maxLR - initLR) * step / warmupSteps;
    else
        % Cosine annealing
        progress = (step - warmupSteps) / (totalSteps - warmupSteps);
        lr = finalLR + 0.5 * (maxLR - finalLR) * (1 + cos(pi * progress));
    end
end

function plotTrainingCurves(history)
% Plot training history
    figure('Position', [100, 100, 1200, 1000], 'Color', 'w');
    
    % Loss curves
    subplot(2, 2, 1);
    semilogy(history.trainLoss, 'LineWidth', 1.5); hold on;
    semilogy(history.valLoss, 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Loss');
    title('Training Curves');
    legend('Train', 'Validation');
    grid on;
    
    % Learning rate
    subplot(2, 2, 2);
    plot(history.lr, '.-', 'Color', [0.5, 0, 0.5], 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Learning Rate');
    title('Learning Rate Schedule');
    grid on;
    
    % R² evolution
    subplot(2, 2, 3);
    vars = {'c_s', 'c_e', '\phi_e', '\phi_s', 'J_{Li}'};
    epochs = 1:size(history.r2, 1);
    for i = 1:5
        plot(epochs, history.r2(:, i), '.-', 'DisplayName', vars{i}, 'LineWidth', 1.2);
        hold on;
    end
    yline(0.98, 'r--', 'LineWidth', 1, 'DisplayName', 'Target R²=0.98');
    xlabel('Epoch'); ylabel('R² Value');
    title('R² Evolution');
    legend('Location', 'best');
    grid on;
    
    % RMSE evolution (concentrations)
    subplot(2, 2, 4);
    % Note: R² is plotted instead of RMSE here since RMSE requires 
    % storing per-epoch metrics. Can be extended.
    plot(epochs, 1 - history.r2(:, 1), '.-', 'DisplayName', 'c_s', 'LineWidth', 1.2); hold on;
    plot(epochs, 1 - history.r2(:, 2), '.-', 'DisplayName', 'c_e', 'LineWidth', 1.2);
    set(gca, 'YScale', 'log');
    xlabel('Epoch'); ylabel('1 - R² (log scale)');
    title('Concentration Error Evolution');
    legend('Location', 'best');
    grid on;
    
    sgtitle('PINN-FHM Training Report', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Save figure
    saveas(gcf, 'training_report.svg');
    saveas(gcf, 'training_report.png');
    fprintf('Training curves saved.\n');
end
