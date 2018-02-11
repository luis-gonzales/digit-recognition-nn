function [X_train, y_train, X_CV, y_CV, X_test, y_test] = splitData(X,y)
% Splits X and y into 60/20/20

% Retrieve indexing variables
nTrain = round(size(X,1) * 0.6);
nCV    = round(size(X,1) * 0.2);

X_train = X(1:nTrain,:);
y_train = y(1:nTrain);

X_CV = X(nTrain+1 : nTrain+nCV,:);
y_CV = y(nTrain+1 : nTrain+nCV);

X_test = X(nTrain+nCV+1:end,:);
y_test = y(nTrain+nCV+1:end);

end