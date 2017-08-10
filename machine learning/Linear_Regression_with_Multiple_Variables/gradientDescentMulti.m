function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %s = zeros(length(theta),1);
    %for j=1:length(theta)
    %  for i=1:m
    %    s(j) = s(j) + (alpha/m)*(X(i,:) * theta - y(i)) * X(i,j);
    %  end;
    %end
    %theta = theta - s;

    i=1;s1=0;s2=0;s3=0;
    while(i<=m)
      s1 = s1 + (alpha/m)*(theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3) - y(i)) * X(i,1);
      s2 = s2 + (alpha/m)*(theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3) - y(i)) * X(i,2);
      s3 = s3 + (alpha/m)*(theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3) - y(i)) * X(i,3);     
      i++;
    endwhile;
    theta(1) = theta(1) - s1;
    theta(2) = theta(2) - s2;
    theta(3) = theta(3) - s3;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
