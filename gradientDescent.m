function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
data=load('ex1data1.txt');
y = data(:,1);
m = length(y); % number of training examples
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
num_iters = 1500;
alpha = 0.01;

J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    X = data(:,2);
	sq = theta(1)+theta(2).*X;
	theta(1) = theta(1) - (1/m)*alpha*sum(sq-y);
	theta(2) = theta(2) - (1/m)*alpha*sum((sq-y).*X);
	J_history(iter) = computeCost(X, y, theta);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
    
end
disp(min(J_history));
end
