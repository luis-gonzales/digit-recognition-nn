function [J, grad] = nnCostFunction(nn_params, ...
                                    input_layer_size, ...
                                    hidden_layer_size, ...
                                    num_labels, ...
                                    X, y, lambda)
% [J grad] = nnCostFunction(nn_params, hidden_layer_size, num_labels, ...
% X, y, lambda) computes the cost, J, and gradient, grad, of the neural
% network. The parameters for the neural network are "unrolled" into the
% vector nn_params and need to be converted back into the weight matrices. 
% 
% The returned parameter grad should be a "unrolled" vector of the
% partial derivatives of the neural network.

% Reshape nn_params back into Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X]; % add bias term
         
% Prediction array
Z2 = Theta1 * X';
A2 = sigmoid(Z2);
A2 = [ones(1, m); A2];
Z3 = Theta2 * A2;
A3 = sigmoid(Z3);      % ith col is the prediction vector for ith sample

% Vectorize y
I_matrix = eye(num_labels);
y_vec = I_matrix(:,y); % (e.g., first sample = 2 -> y_vec(:,1) = [0 1 0 0...]')

% Compute non-regularized cost
J = 0;
for k = 1:num_labels
    y_k = y_vec(k,:)';
    h_k = A3(k,:)';
    J = J + (-y_k'*log(h_k) - (1-y_k)'*log(1-h_k))/m;
end

% Add regularization to cost
J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Calculate partials of Theta1 and Theta2
del_L = A3 - y_vec;
del_2 = (Theta2(:,2:end))' * del_L .* sigmoidGradient(Z2);
Theta1_grad = del_2 * X / m + lambda/m*[zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2_grad = del_L * A2' / m + lambda/m*[zeros(num_labels,1) Theta2(:,2:end)];

% Because X and A2 have include bias term, gradient with respect to these
% bias terms included in the matrix algebra

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end