function [m, P] = FitSelect(P, MDL, K)
% returns top members, sorted from best to worst, and best MDL score
[m, I] = sort(MDL);
m = m(1);
P = P(I(1:K));
end