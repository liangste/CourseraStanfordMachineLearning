function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

#size(X); # 12 x 2
#size(y); # 12 x 1
#size(theta); # 2 x 1
#size(lambda); # 1 x 1

# Cost Calculation
tmp_J = sum(power(X * theta - y, 2));
tmp_theta = sum(power(theta(2:end, :), 2)) * lambda;
J = (tmp_J + tmp_theta) / 2 / m;

# Gradient Calculation
grad_reg = zeros(size(theta));
grad_reg(2:end, :) = lambda .* theta(2:end, :);
grad = (X' * (X * theta - y) + grad_reg) / m;

% =========================================================================

grad = grad(:);

end
