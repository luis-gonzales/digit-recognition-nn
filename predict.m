function p = predict(Theta1, Theta2, X)
% p = predict(Theta1, Theta2, X) outputs the predicted label of X given the
% trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% Initialize p 
p = zeros(size(X, 1), 1);

% Perform forward propagation
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2'); % m by 10

% Output max of each row
[~, p] = max(h2, [], 2);

end