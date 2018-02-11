function [X_scale] = featureScale(X)
% Scale to [-0.5, 0.5]

X_max = max(max(X));
X_min = min(min(X));

X_scale = (X - (X_max-X_min)/2) / (X_max-X_min);

end