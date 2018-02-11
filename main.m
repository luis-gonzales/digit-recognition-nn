% Data obtained from https://www.kaggle.com/c/digit-recognizer
% Only using train.csv since test.csv does not contain labels.
% test.csv is structured such that the first col contains labels, while 
% remaining cols contain unrolled pixels [0-255].

%% Read in and process data
clear,clc,close all

M = csvread('train.csv',1,0);
X = M(:,2:end);
y = M(:,1);
y(y == 0) = 10; % Replace '0' labels with '10' labels

% Split X and y into train/CV/test data (60/20/20)
[X_train, y_train, X_CV, y_CV, X_test, y_test] = splitData(X,y);

% Caution: if imnoise used, X_train mapped to [0,1]
%X_train = imnoise(X_train, 'salt & pepper');
%X_train = imnoise(X_train, 'gaussian');

% Display four arbitrary samples for visualization
displayData(X_train([80 2 12 51], :));

% Map pixels to [-0.5, 0.5], assuming currently in [0, 255]
X_train = featureScale(X_train);
X_CV    = featureScale(X_CV);

%% Setup parameters and train NN with various lambdas

input_layer_size  = 784;  % 28x28 input images
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10

% Setup Initial Theta matrices            
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 350);

% Desired regularization parameters for training
lambda_vec = [0 0.005 0.01 0.02 0.04 0.08 0.16 0.24 0.32 ...
              0.45 0.64 1 1.28 1.6 2 2.56 3.5 4.2 5.12];

% Initialize record-keeping vectors
del_cost = zeros(1, length(lambda_vec));
J_train  = zeros(1, length(lambda_vec));
J_CV     = zeros(1, length(lambda_vec));

for k = 1:length(lambda_vec)
    % Create handle for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, ...
                                       X_train, y_train, lambda_vec(k));

    % Minimize cost and obtain optimal Thetas
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    % Store cost value post-optimization
    J_train(k)   = cost(end);
    [J_CV(k), ~] = nnCostFunction(nn_params, ...
                                  input_layer_size, ...
                                  hidden_layer_size, ...
                                  num_labels, ...
                                  X_CV, y_CV, lambda_vec(k));
    
    % To ensure proper training per lambda, save final del_cost
    del_cost(k) = cost(end-1) - cost(end);
end

% Plot J_train and J_CV vs lambda to capture optimal lambda
figure(2), plot(lambda_vec, J_train, lambda_vec, J_CV, 'LineWidth', 5.5)
legend('J_{train}', 'J_{CV}')
xlabel('{\lambda}'), ylabel('J')
set(gca,'fontsize',35)

% Plot del_cost vs lambda to ensure adequate amount of training steps
figure(3), plot(lambda_vec, del_cost)

%% Use training from above to pick optimal lambda and obtain NN Thetas

lambda = 0.15;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
                 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Use final Thetas to obtain NN performance metrics          

% Map pixels to [-0.5, 0.5] assuming currently in [0, 255]
X_test = featureScale(X_test);

% Capture predictions of X_train and X_test
pred_train = predict(Theta1, Theta2, X_train);
pred_test  = predict(Theta1, Theta2, X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y_train)) * 100);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);