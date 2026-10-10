%% TEST_PIPELINE - End-to-end verification of PINN pipeline
clc;
fprintf('=== Testing Full PINN Pipeline ===\n');

% Battery parameters
bp = batteryParams();
dataPath = fullfile('..', 'Data.mat');
[trainData, valData, testData, ns] = loadData(dataPath);

% Init Network
params = initNetwork(4, 256, 6, 5);

% Pretraining test (1 batch)
fprintf('\n--- Testing Pretraining Step ---\n');
pw = struct('w1', 1.0, 'w2', 1.0, 'w3', 1.0, 'w4', 1.0, 'w5', 1.0);
idx = 1:256;
x_dl = dlarray(trainData.inputs(idx, 1));
t_dl = dlarray(trainData.inputs(idx, 2));
C_dl = dlarray(trainData.inputs(idx, 3));
T_dl = dlarray(trainData.inputs(idx, 4));
cs_d = dlarray(trainData.cs(idx));
ce_d = dlarray(trainData.ce(idx));
phie_d = dlarray(trainData.phie(idx));
phis_d = dlarray(trainData.phis(idx));
JLi_d  = dlarray(trainData.JLi(idx));

[loss_pre, grads_pre] = dlfeval(@pretrainGradients, params, ...
    x_dl, t_dl, C_dl, T_dl, cs_d, ce_d, phie_d, phis_d, JLi_d, pw);

avgGrad = [];
avgSqGrad = [];
[params, avgGrad, avgSqGrad] = adamupdate(params, grads_pre, avgGrad, avgSqGrad, 1, 5e-3);
fprintf('Pretrain batch loss: %.4f - PASSED\n', extractdata(loss_pre));

% Physics training test (1 batch)
fprintf('\n--- Testing Physics Training Step ---\n');
w = struct('w1', 8e-15, 'w2', 1e-14, 'w3', 1e-2, 'w4', 1.0, 'w5', 2.0, ...
           'w6', 1.0, 'w7', 1.0, 'w8', 1.0);

[loss_phys, grads_phys] = dlfeval(@physicsGradients, params, ...
    x_dl, t_dl, C_dl, T_dl, cs_d, ce_d, phie_d, phis_d, JLi_d, ns, bp, w);

avgGrad = [];
avgSqGrad = [];
[params, avgGrad, avgSqGrad] = adamupdate(params, grads_phys, avgGrad, avgSqGrad, 1, 5e-4);
fprintf('Physics batch loss: %.4f - PASSED\n', extractdata(loss_phys));

% Validation metrics test
fprintf('\n--- Testing Validation Metrics ---\n');
v = computeValMetrics(params, valData.inputs(1:500,:), ...
    valData.cs(1:500), valData.ce(1:500), valData.phie(1:500), ...
    valData.phis(1:500), valData.JLi(1:500));
fprintf('Val total loss: %.4f - PASSED\n', v.totalLoss);

% Evaluation and plotting test
fprintf('\n--- Testing Evaluation and Plotting ---\n');
evaluateResults(params, ns, bp, dataPath, 298.0, 1);
fprintf('Evaluation plots generated - PASSED\n');

fprintf('\n=====================================\n');
fprintf('ALL PIPELINE COMPONENTS VERIFIED OK!\n');
fprintf('=====================================\n');
