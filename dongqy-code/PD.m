function result = PD(X)
[~,p] = chol(X);
result = (~p);
end