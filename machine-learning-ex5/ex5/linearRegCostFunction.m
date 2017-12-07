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

% --- cost function --- %

% calculate hypothesis, or prediction
h = X * theta;
% define number of features
n = size(theta);
% calculate cost function of regularized linear regression, excluding theta0 term
J = 1/(2*m)*sum((h-y).^2) + lambda/(2*m)*sum(theta(2:n).^2);

% --- gradient --- %

grad = (1/m) * X' * (h-y);  % for all partial derivatives
grad(2:n) = grad(2:n) + (lambda/m)*theta(2:n);  % add regularized parameter for j>=1 terms

% =========================================================================

grad = grad(:);

end
