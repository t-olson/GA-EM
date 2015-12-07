classdef Individual < handle        % !!!Caution : handle subclass, pass by reference
    properties
        Binary      % Boolean array of Mmax length
        Mu          % d x Mmax matrix, each column represents one mu_m
        Sigma       % (d x d) x Mmax matrix, each layer represents one sigma_m
        weight      % 1 x Mmax array, sum should be 1
        best_ind    % indicator if the individual is currently the best
        mdl         % The minimum description length (MDL) for this individual
    end
    methods
        % Constructor
        function obj = Individual(Mmax, d, X)
            %   Mmax : int, > 0
            %   d : dimension of trainning data
            if nargin == 2
                obj.Binary = zeros(1, Mmax);
                obj.Mu = zeros(d, Mmax);
                obj.Sigma = repmat(zeros(d), [1 1 Mmax]);
                obj.weight = zeros(1, Mmax);
                obj.best_ind = false;
                obj.mdl = Inf;
                
                assert(length(obj.Binary) == size(obj.Mu, 2));
                assert(length(obj.Binary) == size(obj.Sigma, 3));
                assert(size(obj.Mu, 1) == size(obj.Sigma, 1));
            % Deep Copy constructor (because Individual's are passed by ref)
            %   Mmax : a Individual object
            elseif nargin == 1 && isa(Mmax, 'Individual')
                obj.Binary = Mmax.Binary;
                obj.Mu = Mmax.Mu;
                obj.Sigma = Mmax.Sigma;
                obj.weight = Mmax.weight;
                obj.best_ind = Mmax.best_ind;
                obj.mdl = Mmax.mdl;
            % Default constructor
            elseif nargin == 0
            % Random initialization
            elseif nargin == 3
                obj.Binary = zeros(1, Mmax);
                obj.Mu = zeros(d, Mmax);
                obj.Sigma = repmat(zeros(d), [1 1 Mmax]);
                obj.weight = zeros(1, Mmax);
                obj.best_ind = false;
                
                assert(length(obj.Binary) == size(obj.Mu, 2));
                assert(length(obj.Binary) == size(obj.Sigma, 3));
                assert(size(obj.Mu, 1) == size(obj.Sigma, 1));
                
                N = size(X, 2);
                M = randi([1,Mmax]);                % a random number as initial M
                p = randperm(Mmax, M);              % randomly select M components
                sample_cov = X*X.'/N;
                for i = 1:M
                    obj.Binary(p(i)) = 1;
                    obj.Mu(:, p(i)) = X(:, randi([1, N]));
                    obj.Sigma(:, :, p(i)) = sample_cov;
                    obj.weight(p(i)) = 1/M;
                end
                obj.mdl = obj.MDL(X);
            else
                error('Constructor error.');
            end
        end
        
        
        % Return the number of components M in this individual
        function M = num(obj)
            M = sum(obj.Binary);
        end
        
        % Perform R EM steps for fitting the Gaussian mixture model 
        % using specific individual.
        % !!!Caution : obj changed
        %   obj : individual
        %   X : d x n data matrix
        %   R : number of iterations
        function EM_gmm(obj, X, R)
            N = size(X, 2);
            M = obj.num();
            mu = obj.Mu(:, obj.Binary == 1);
            sigma = obj.Sigma(:,:,obj.Binary == 1);
            w = obj.weight(obj.Binary == 1);
            new_mu = mu;
            new_sigma = sigma;
            new_w = w;
            assert(M == size(mu, 2));
            assert(M == size(sigma, 3));
            assert(M == length(w));

            maxiter = R;                % maximum num of iterations
            gamma = zeros(M, N);        % expectation of the indicator variable \gamma_i,k

            % R steps of EM Algorithm
            for iter = 1:maxiter

                % E-step
                for k = 1:M
                    gamma(k,:) = w(k).*mvnpdf(X', mu(:, k)', sigma(:,:,k));
                end
                gamma = gamma./repmat(sum(gamma, 1),M,1);

                % M-step
                for k = 1:M
                    % Optimize w_k
                    new_w(k) = mean(gamma(k,:));
                    % Optimize mu_k
                    new_mu(:, k) = sum(bsxfun(@times, gamma(k, :), X), 2)./sum(gamma(k, :));
                    % Optimize sigma_k
                    centered_X = bsxfun(@minus, X, new_mu(:,k));
                    new_sigma(:,:,k) = (bsxfun(@times, gamma(k, :), centered_X)*centered_X.')./sum(gamma(k, :));
                end

                mu = new_mu;
                sigma = new_sigma;
                w = new_w;

            end

            obj.Mu(:, obj.Binary == 1) = new_mu;
            obj.Sigma(:,:,obj.Binary == 1) = new_sigma;
            obj.weight(obj.Binary == 1) = new_w;
            obj.mdl = obj.MDL(X);
        end
        
        % Compute the log-likelihood of data X based on this individual
        %   X : d x n training data matrix
        function llh = log_likelihood(obj, X)
            mu = obj.Mu(:, obj.Binary == 1);
            sigma = obj.Sigma(:,:,obj.Binary == 1);
            w = obj.weight(obj.Binary == 1);
            
            N = size(X, 2);
            M = size(mu, 2);                    % the number of components M in this individual 
            assert(M == obj.num());
%           
            if M~=0
                tmp = 0;
                for k = 1:M
                    tmp = tmp + w(k).*mvnpdf(X', mu(:, k)', sigma(:,:,k));
                end

                llh = sum(log(tmp));
            else
                llh = -Inf;
            end
        end
        
        % Compute the MDL of data X based on this individual
        %   X : d x n training data matrix
        function r = MDL(obj, X)
            d = size(X, 1);
            N = size(X, 2);
            L = d + d*(d+1)/2;
            r = -log_likelihood(obj, X) + obj.num()*(L+1)/2*log(N);
        end
        
        % Do the enforced mutation on this individual
        %   !!!Caution : obj changed
        %   X : d x n training data matrix
        %   t : correlation threshold
        function enforced_mutation(obj, X, t)
            M = obj.num();
            N = size(X, 2);
            mu = obj.Mu(:, obj.Binary == 1);
            sigma = obj.Sigma(:,:,obj.Binary == 1);
            w = obj.weight(obj.Binary == 1);
            % expectation of the indicator variable \gamma_i,k (posterior probability, z^i_m in the paper context)
            gamma = zeros(M, N);        
            for k = 1:M
                gamma(k,:) = w(k).*mvnpdf(X', mu(:, k)', sigma(:,:,k));
            end
            gamma = gamma./repmat(sum(gamma, 1),M,1);
            r = corrcoef(gamma.');                      % correlation coefficients r_jk
            r = r - diag(diag(r));                      % get ride of diagonal elements (1's)
            mutate_cand = abs(r) > t | isnan(r);
            [row, col] = find(mutate_cand);
            upper_part = row > col;
            cand_row = row(upper_part);
            cand_col = col(upper_part);
            assert(length(cand_row) == length(cand_col));
            dice = randi([0,1],1,length(cand_row));
            cand = [cand_row(dice==1)', cand_col(~(dice==1))'];
            indices = find(obj.Binary == 1);
            cand_ind = indices(cand);
            assert(length(cand_ind) == length(cand));
            sample_cov = X*X.'/N;
            for i = 1:length(cand_ind)
                if obj.Binary(cand_ind(i)) == 1
                    obj.Binary(cand_ind(i)) = 0;
                else
                    obj.Binary(cand_ind(i)) = 1;
                    obj.Mu(:, cand_ind(i)) = X(:, randi([1, N]));
                    obj.Sigma(:, :, cand_ind(i)) = sample_cov;
                end
            end
            % Deal with weight after the enforced mutation
            new_M = obj.num();
            obj.weight(obj.Binary == 1) = 1/new_M;
            obj.mdl = obj.MDL(X);
        end
        
        % Do the mutation on this individual
        %   !!!Caution : obj changed
        %   X : d x n training data matrix
        %   pm : mutation probability
        function mutation(obj, X, pm)
            Mmax = length(obj.Binary);
            d = size(X, 1);
            N = size(X, 2);
            L = d + d*(d+1)/2;
            upper_bound = max(X, [], 2);
            lower_bound = min(X, [], 2);
            mutated = rand(1, Mmax) <= pm;
            indices = 1:Mmax;
            mutated_ind = indices(mutated);
            sample_cov = X*X.'/N;
            for i = mutated_ind
                if obj.Binary(i) == 1
                    obj.Binary(i) = 0;
                else
                    obj.Binary(i) = 1;
                    dice = rand(d, 1);
                    tmp = lower_bound + (upper_bound - lower_bound).*dice;
                    obj.Mu(dice <= pm/L, i) = tmp(dice <= pm/L);
                    obj.Sigma(:, :, i) = sample_cov;
                end
            end
            % Deal with weight after the mutation
            new_M = obj.num();
            obj.weight(obj.Binary == 1) = 1/new_M;
            obj.mdl = obj.MDL(X);
        end
    end
end
