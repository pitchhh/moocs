function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

C_vals = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vals = [0.01 0.03 0.1 0.3 1 3 10 30];

% compute prediction error for every pair of C and sigma values
for i=1:length(C_vals)
    for j=1:length(sigma_vals)
        C_value = C_vals(i);
        sigma_value = sigma_vals(j);
        % define model in a similar fashion from ex6
        model = svmTrain(X, y, C_value, @(x1, x2) gaussianKernel(x1, x2, sigma_value));
        % predict labels on the cross validation set (Xval, yval)
        predictions = svmPredict(model, Xval);
        prediction_errors(i, j) = mean(double(predictions ~= yval));
    end
end

% find the pair in prediction_errors that computes the smallest prediction error
min_error = min(min(prediction_errors));
% use min() twice, first find minimum from each column, then minimum of each row
[i j] = find(prediction_errors == min_error);
C = C_vals(i);
sigma = sigma_vals(j);

% =========================================================================

end
