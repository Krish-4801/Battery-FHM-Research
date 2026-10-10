function original = zscoreDenormalize(normalized, mu, sigma)
% ZSCOREDENORMALIZE Reverse z-score normalization to recover original scale
%   original = zscoreDenormalize(normalized, mu, sigma)
%
% Args:
%   normalized - Normalized data (numeric or dlarray)
%   mu         - Mean used during normalization
%   sigma      - Standard deviation used during normalization
%
% Returns:
%   original - Data in original scale: normalized * sigma + mu

    original = normalized .* sigma + mu;
end
