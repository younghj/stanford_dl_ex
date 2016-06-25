function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
  po = false;
  if exist('pred_only','var')
    po = pred_only;
  end;

  %% reshape into network
  stack = params2stack(theta, ei);
  numHidden = numel(ei.layer_sizes) - 1;
  hAct = cell(numHidden+1, 1);
  gradStack = cell(numHidden+1, 1);
  %% forward prop
  %%% YOUR CODE HERE %%%

  struct_levels_to_print(0)

  a = data;
  hAct{1} = data;

  for i=1:numHidden+1
    curr_stack = stack{i,1};
    W = curr_stack.W;
    b = curr_stack.b;
    z = bsxfun(@plus, W * a, b);

    %a = bsxfun(@rdivide, exp(z), sum(exp(z),2));



    if i == numHidden+1
      a = exp(z);
      a = bsxfun(@rdivide, a, sum(a,1));
    else
      switch ei.activation_fun
      case 'logistic'
        a = sigmoid(z);
      case 'tanh'
        a = tanh(z);
      case 'relu'
        a = relu(z);
      end
    end

    hAct{i+1} = a;
  end

  %y_hat = exp(hAct{end});
  %pred_prob = bsxfun(@rdivide, y_hat, sum(y_hat, 1));
  %hAct{end} = pred_prob;

  pred_prob = hAct{end};

  %pred_prob = hAct{end};

  %% return here if only predictions desired.
  if po
    cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
    grad = [];  
    return;
  end;

  %% compute cost
  %%% YOUR CODE HERE %%%

  A = log(pred_prob);
  I = sub2ind(size(A), labels', 1:size(A,2));

  ceCost = -sum(A(I));

  %% compute gradients using backpropagation
  %%% YOUR CODE HERE %%%
  deltaStack = cell(numHidden+1, 1);

  eqa = zeros(size(pred_prob));
  eqa(I) = 1;
  deltaStack{end+1} = pred_prob - eqa;

  for l=numHidden+1:-1:2
    W = stack{l,1}.W;
    a = hAct{l};

    switch ei.activation_fun
    case 'logistic'
      f_prime = a.*(1-a);
    case 'tanh'
      f_prime = 1-(a.^2);
    case 'relu'
      f_prime = a>0;
    end

    deltaStack{l} = (W'*deltaStack{l+1}).*f_prime;
  end

  for l=1:numHidden+1
    gradStack{l}.W = deltaStack{l+1}*hAct{l}';
    gradStack{l}.b = sum(deltaStack{l+1},2);
  end


  %% compute weight penalty cost and gradient for non-bias terms
  %%% YOUR CODE HERE %%%

  wCost = 0;

  for l=1:numHidden+1
    wCost = wCost + ei.lambda/2 * sum(stack{l}.W(:) .^ 2);
  end

  cost = ceCost + wCost;

  for l = numHidden: -1 :1
    gradStack{l}.W = gradStack{l}.W + ei.lambda* stack{l}.W;
  end

  %% reshape gradients into vector
  [grad] = stack2params(gradStack);
end



