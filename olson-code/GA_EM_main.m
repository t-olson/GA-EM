function GA_EM_main()
close all;
SEED = 123; % change this to get different data sets

% Set GA-EM Parameters
R = 3; % number of EM steps for each GA iteration
M = 15; % max number of components
K = 6; % size of parent population
H = 4;%floor(.8*K); % number of offspring
p_m = 0.02; % mutation rate

rng(SEED); % initialize sample data
C = 5; % actual number of clusters
N = 300 * C; % number of points
d = 2; % dimensions

[data, means, sigmas, MDL_true] = SampleData(N,C,d); % generate data
disp(['True MDL: ', num2str(MDL_true)]); % print MDL value

% plot actual data
scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/6 scrsz(3)*2/3 scrsz(4)*2/3])
subplot(2,2,1);
scatter(data(:,1), data(:,2));
hold on;
scatter(means(:,1),means(:,2),'X');

% plot true mixture (draw contour at f(x,y) = 0.1)
xRange = min(data(:,1))-1:.1:max(data(:,1))+1; % x axis
yRange = min(data(:,2))-1:.1:max(data(:,2))+1; % y axis
[X, Y] = meshgrid(xRange,yRange); % all combinations of x, y
for i=1:C
    Z = mvnpdf([X(:) Y(:)], means(i,:), sigmas(:,:,i)); % compute pdf
    Z = reshape(Z,size(X));
    contour(X,Y,Z,[.1,.1], 'LineColor', [0 0 0], 'LineWidth', 2);  % contour plot
end
title(['Actual data and mixtures, MDL = ',  num2str(MDL_true)]);

% Run EM algorithm for comparison
rng(SEED); % reset random seed to ensure reproducibility
disp('Running EM alone');
tic
Pop = InitPopulation(data, 15, C); % initialize (requires same M and K >= C to match GA-EM init)
EM_result = Pop(C); % only use the C^th one, which has correct components enabled
EM_result = EM(EM_result, data, Inf);
MDL_EM = MDLencode(EM_result,data); % compute MDL value for EM result
toc
disp(['EM: ', num2str(MDL_EM)]); % print MDL value

% plot EM-mixture
subplot(2,2,2);
scatter(data(:,1), data(:,2));
hold on;
for i=1:C
    Z = mvnpdf([X(:) Y(:)], EM_result.means(:,i)', EM_result.covs(:,:,i)); % compute Gaussian pdf
    Z = reshape(Z,size(X));
    contour(X,Y,Z,[.1,.1], 'LineColor', [0 1 0], 'LineWidth', 2);  % contour plot
end
title(['Best EM Mixture, MDL = ',  num2str(MDL_EM)]);

% Run GA_EM algorithm (plots of fits and MDL vs iteration are generated
% internally)
rng(SEED); % reset random seed
disp('Running GA-EM');
tic
[GA_EM_result, MDL_list] = GA_EM(data, R, M, K, H, p_m);
toc
disp(['GA_EM: ', num2str(MDL_list(end))]); % print final MDL value

% plot GA_EM-mixture
subplot(2,2,3);
scatter(data(:,1), data(:,2));
hold on;
index = 1:M;
index = index(logical(GA_EM_result(1).code));
for i=1:length(index)
    Z = mvnpdf([X(:) Y(:)], GA_EM_result.means(:,index(i))', GA_EM_result.covs(:,:,index(i))); % compute Gaussian pdf
    Z = reshape(Z,size(X));
    contour(X,Y,Z,[.1,.1], 'LineColor', [0 1 0], 'LineWidth', 2);  % contour plot
end
title(['Best EM Mixture, MDL = ',  num2str(MDL_list(end))]);

% show MDL plot
subplot(2,2,4);
hold off;
plot(1:length(MDL_list), MDL_list(1:length(MDL_list)))
title('MDL vs iterations');
end
