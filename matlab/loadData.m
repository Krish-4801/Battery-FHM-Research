function [trainData, valData, testData, ns] = loadData(dataPath)
% LOADDATA Load Data.mat, split into train/val/test (70/15/15), and normalize
%
% Args:
%   dataPath - Full or relative path to Data.mat
%
% Returns:
%   trainData - Struct with fields (inputs, cs, ce, phie, phis, JLi) [all single]
%   valData   - Struct with fields (inputs, cs, ce, phie, phis, JLi) [all single]
%   testData  - Struct with fields (inputs, cs, ce, phie, phis, JLi) [all single]
%   ns        - Normalization statistics computed ONLY from training partition

    if nargin < 1 || isempty(dataPath)
        dataPath = 'Data.mat';
    end

    fprintf('Loading preprocessed dataset from %s ...\n', dataPath);
    loaded = load(dataPath);

    % Verify presence of variables
    reqVars = {'inputs_all', 'cs_all', 'ce_all', 'phie_all', 'phis_all', 'JLi_all'};
    for i = 1:length(reqVars)
        if ~isfield(loaded, reqVars{i})
            error('Variable %s not found in %s', reqVars{i}, dataPath);
        end
    end

    inputs_all = double(loaded.inputs_all);
    cs_all     = double(loaded.cs_all);
    ce_all     = double(loaded.ce_all);
    phie_all   = double(loaded.phie_all);
    phis_all   = double(loaded.phis_all);
    JLi_all    = double(loaded.JLi_all);

    N = size(inputs_all, 1);
    fprintf('Total samples loaded: %d\n', N);

    % Replicate train_test_split(test_size=0.3, random_state=42) followed by
    % train_test_split(test_size=0.5, random_state=42)
    % Consistent seed and random permutation:
    rng(42);
    shuffled_indices = randperm(N);

    % Split sizes: 70% train, 15% validation, 15% test
    n_train = round(0.70 * N);
    n_temp  = N - n_train;
    n_val   = round(0.50 * n_temp);
    
    train_idx = shuffled_indices(1:n_train);
    val_idx   = shuffled_indices(n_train + 1 : n_train + n_val);
    test_idx  = shuffled_indices(n_train + n_val + 1 : end);

    % Extract subsets
    train_inputs_raw = inputs_all(train_idx, :);
    train_cs_raw     = cs_all(train_idx, :);
    train_ce_raw     = ce_all(train_idx, :);
    train_phie_raw   = phie_all(train_idx, :);
    train_phis_raw   = phis_all(train_idx, :);
    train_JLi_raw    = JLi_all(train_idx, :);

    val_inputs_raw   = inputs_all(val_idx, :);
    val_cs_raw       = cs_all(val_idx, :);
    val_ce_raw       = ce_all(val_idx, :);
    val_phie_raw     = phie_all(val_idx, :);
    val_phis_raw     = phis_all(val_idx, :);
    val_JLi_raw      = JLi_all(val_idx, :);

    test_inputs_raw  = inputs_all(test_idx, :);
    test_cs_raw      = cs_all(test_idx, :);
    test_ce_raw      = ce_all(test_idx, :);
    test_phie_raw    = phie_all(test_idx, :);
    test_phis_raw    = phis_all(test_idx, :);
    test_JLi_raw     = JLi_all(test_idx, :);

    % Compute normalization statistics strictly from training data
    ns = struct();
    ns.x_mean = mean(train_inputs_raw(:, 1));
    ns.x_std  = std(train_inputs_raw(:, 1));

    ns.t_mean = mean(train_inputs_raw(:, 2));
    ns.t_std  = std(train_inputs_raw(:, 2));

    ns.C_mean = mean(train_inputs_raw(:, 3));
    ns.C_std  = std(train_inputs_raw(:, 3));

    ns.T_mean = mean(train_inputs_raw(:, 4));
    ns.T_std  = std(train_inputs_raw(:, 4));

    ns.cs_mean = mean(train_cs_raw);
    ns.cs_std  = std(train_cs_raw);

    ns.ce_mean = mean(train_ce_raw);
    ns.ce_std  = std(train_ce_raw);

    ns.phie_mean = mean(train_phie_raw);
    ns.phie_std  = std(train_phie_raw);

    ns.phis_mean = mean(train_phis_raw);
    ns.phis_std  = std(train_phis_raw);

    ns.JLi_mean = mean(train_JLi_raw);
    ns.JLi_std  = std(train_JLi_raw);

    % Apply z-score normalization to all sets
    trainData = struct();
    trainData.inputs = single(normalizeInputs(train_inputs_raw, ns));
    trainData.cs     = single(zscoreNormalize(train_cs_raw, ns.cs_mean, ns.cs_std));
    trainData.ce     = single(zscoreNormalize(train_ce_raw, ns.ce_mean, ns.ce_std));
    trainData.phie   = single(zscoreNormalize(train_phie_raw, ns.phie_mean, ns.phie_std));
    trainData.phis   = single(zscoreNormalize(train_phis_raw, ns.phis_mean, ns.phis_std));
    trainData.JLi    = single(zscoreNormalize(train_JLi_raw, ns.JLi_mean, ns.JLi_std));

    valData = struct();
    valData.inputs = single(normalizeInputs(val_inputs_raw, ns));
    valData.cs     = single(zscoreNormalize(val_cs_raw, ns.cs_mean, ns.cs_std));
    valData.ce     = single(zscoreNormalize(val_ce_raw, ns.ce_mean, ns.ce_std));
    valData.phie   = single(zscoreNormalize(val_phie_raw, ns.phie_mean, ns.phie_std));
    valData.phis   = single(zscoreNormalize(val_phis_raw, ns.phis_mean, ns.phis_std));
    valData.JLi    = single(zscoreNormalize(val_JLi_raw, ns.JLi_mean, ns.JLi_std));

    testData = struct();
    testData.inputs = single(normalizeInputs(test_inputs_raw, ns));
    testData.cs     = single(zscoreNormalize(test_cs_raw, ns.cs_mean, ns.cs_std));
    testData.ce     = single(zscoreNormalize(test_ce_raw, ns.ce_mean, ns.ce_std));
    testData.phie   = single(zscoreNormalize(test_phie_raw, ns.phie_mean, ns.phie_std));
    testData.phis   = single(zscoreNormalize(test_phis_raw, ns.phis_mean, ns.phis_std));
    testData.JLi    = single(zscoreNormalize(test_JLi_raw, ns.JLi_mean, ns.JLi_std));

    fprintf('Data successfully split and normalized:\n');
    fprintf('  Train: %d samples\n', size(trainData.inputs, 1));
    fprintf('  Val:   %d samples\n', size(valData.inputs, 1));
    fprintf('  Test:  %d samples\n', size(testData.inputs, 1));
end

function norm_in = normalizeInputs(raw_in, ns)
    norm_in = zeros(size(raw_in));
    norm_in(:, 1) = zscoreNormalize(raw_in(:, 1), ns.x_mean, ns.x_std);
    norm_in(:, 2) = zscoreNormalize(raw_in(:, 2), ns.t_mean, ns.t_std);
    norm_in(:, 3) = zscoreNormalize(raw_in(:, 3), ns.C_mean, ns.C_std);
    norm_in(:, 4) = zscoreNormalize(raw_in(:, 4), ns.T_mean, ns.T_std);
end
