function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.

% initialize distance matrix of size m x k to store distance between each point to each centroid
distance = zeros(size(X,1), K);

% loop over K centroids
for k=1:K
    % compare relative square distance from example point to each centroid
    distance(:,k) = sum(bsxfun(@minus, X, centroids(k,:)).^2,2);
% compute min values and indexes for each pair in distance vector
% min returns a row vector with min value of each column, so we should use transpose of distance matrix
[minVal, minIndex] = min(distance');
% then transpose indexes back to store in idx
idx = minIndex';
end

% =============================================================

end
