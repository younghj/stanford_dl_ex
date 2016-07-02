%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
wX = W*x;
val = W'*wX-x;
m = size(x,2);

sparse = params.lambda*sum(sqrt(wX.^2+params.epsilon)(:));
tempcost = 0.5*norm(val,'fro').^2/m;
%tempcost = 0.5*sqrt(sum((val.^2)(:)))/m;
cost = sparse + tempcost;

gradW = 2*W*val*x';
gradWT = 2*wX*val';
reg = params.lambda *(wX./sqrt(wX.^2+params.epsilon))*x';
Wgrad = reg + (gradW + gradWT)/(2*m);

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
