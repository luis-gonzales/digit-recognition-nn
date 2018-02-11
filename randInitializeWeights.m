function W = randInitializeWeights(L_in, L_out)
% W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
% of a layer with L_in incoming connections and L_out outgoing 
% connections. Randomness breaks symmetry and helps with subsequent training.

epsilon = sqrt(6)/sqrt(L_in+L_out);

% size(W) = [L_out, 1+L_in] where '1+' represents bias term
W = rand(L_out, 1 + L_in)*2*epsilon - epsilon;

end