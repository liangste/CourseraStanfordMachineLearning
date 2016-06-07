function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

m; # 5000, the number of training examples
input_layer_size; # 400
hidden_layer_size; # 25
num_labels; #10, output layer size
size(X); # 5000 x 400
size(Theta1); # 25 x 401
size(Theta2); # 10 x 26
size(y); # 5000 x 1

delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for i = 1 : m
  # forward propagation
  a_1 = [1, X(i, :)]; # 1 x 401
  z_2 = Theta1 * a_1'; # 25 x 1
  a_2 = [1; sigmoid(z_2)]; # 26 x 1
  z_3 = Theta2 * a_2; # 10 x 1
  a_3 = sigmoid(z_3); # 10 x 1
  h_theta = a_3; # 10 x 1
  
  # set up temporary outcomes for all labels
  y_k = 1 : num_labels;
  y_k = (y_k == y(i))';
  
  # backpropagation calculations
  del_3 = a_3 - y_k; # 10 x 1
  del_2 = Theta2' * del_3 .* sigmoidGradient([1; z_2]); # 26 x 1
  del_2 = del_2(2:end); # 25 x 1
    
  delta_2 += del_3 * a_2'; # 10 x 26
  delta_1 += del_2 * a_1; # 25 x 401
  
  # compute costs for all labels for this training example
  tmp_J = y_k .* log(h_theta) + (1 - y_k) .* log(1 - h_theta);
  
  # add training example cost to global cost
  J += sum(tmp_J);
endfor

# input to hidden layer regularization

# regularization from Theta1
Theta1_tmp = Theta1(:, 2 : end); # omit bias
Theta1_tmp = power(Theta1_tmp, 2);
Reg_Theta1 = sum(Theta1_tmp, 1);
Reg_Theta1 = sum(Reg_Theta1);

# regularization from Theta2
Theta2_tmp = Theta2(:, 2 : end); # omit bias
Theta2_tmp = power(Theta2_tmp, 2);
Reg_Theta2 = sum(Theta2_tmp, 1);
Reg_Theta2 = sum(Reg_Theta2);

reg = (Reg_Theta1 + Reg_Theta2 ) * lambda / 2 / m;

% -------------------------------------------------------------

J = J / (-m) + reg;
Theta1_grad = delta_1 / m;
Theta2_grad = delta_2 / m;

# gradient regularization

Theta1_grad += [zeros(hidden_layer_size, 1), Theta1(:, 2:end)] * lambda/m;
Theta2_grad += [zeros(num_labels, 1), Theta2(:, 2:end)] * lambda/m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
