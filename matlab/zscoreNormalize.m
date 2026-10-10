function normalized = zscoreNormalize(data, mu, sigma)
% ZSCORENORMALIZE Apply z-score (standardization) normalization
%   normalized = zscoreNormalize(data, mu, sigma)
%
% Args:
%   data   - Input data (numeric or dlarray)
%   mu     - Mean of the training data
%   sigma  - Standard deviation of the training data
%
% Returns:
%   normalized - Z-score normalized data: (data - mu) / (sigma + eps)

    normalized = (data - mu) ./ (sigma + 1e-8);
end
