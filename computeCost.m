function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
data=load('ex1data1.txt');
y = data(:,2);
m = length(y); % number of training examples
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% You need to return the following variables correctly 
sq = (X*theta - y).^2;
J = (1/(2*m))* sum(sq);

end
