function g = sigmoidGradient(z)
% g = sigmoidGradient(z) computes the gradient of the sigmoid function
% evaluated at z (where z can be a matrix, vector, or scalar).

g = sigmoid(z).*(1-sigmoid(z));

end