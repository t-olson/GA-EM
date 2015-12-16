% function GA_EM_main3(varargin)
clearvars -except varargin;

% Set GA-EM Parameters
% R = 10; % number of EM steps for each GA iteration
% M = 20; % max number of components
% K = 6; % size of parent population
% H = 4;%floor(.8*K); % number of offspring
% p_m = 0.1; % mutation rate
R = 3; % number of EM steps for each GA iteration
M = 15; % max number of components
K = 6; % size of parent population
H = 4;%floor(.8*K); % number of offspring
p_m = 0.02; % mutation rate

SEED = 999; % change this to get different data sets & initialization
% if (nargin > 0)
%     SEED = varargin{1};
% end
rng(SEED); % initialize sample data
generateData = false;

if(generateData)
    C = 5; % actual number of clusters
    N = 300 * C; % number of points
    d = 2; % dimensions
    [data, means, sigmas, MDL_true] = SampleData(N,C,d); % generate data
else
    % write your own wrapper like LoadData() to load your true data set
    [data, means, sigmas, C, N, d, MDL_true] = LoadData('..\..\datasets\wine_pca_3'); % real data
end

disp(['True MDL: ', num2str(MDL_true)]); % print MDL value

% plot actual data
scrsz = get(groot,'ScreenSize');
fig=figure('Position',[1 scrsz(4)/6 scrsz(3)*2/3 scrsz(4)*2/3]);
fig.PaperPositionMode = 'auto';
subplot(2,2,1);
scatter3(data(:,1), data(:,2), data(:,3),'filled');
hold on;
scatter3(means(:,1),means(:,2), means(:,3),'X');

% plot true mixture (draw contour at f(x,y) = 0.1)
minX = min(data(:,1));
maxX = max(data(:,1));
minY = min(data(:,2));
maxY = max(data(:,2));
minZ = min(data(:,3));
maxZ = max(data(:,3));
xRange = minX-(maxX-minX)/10:(maxX-minX)/20:maxX+(maxX-minX)/10; % x axis
yRange = minY-(maxY-minY)/10:(maxY-minY)/20:maxY+(maxY-minY)/10; % y axis
zRange = minZ-(maxZ-minZ)/10:(maxZ-minZ)/20:maxZ+(maxZ-minZ)/10; % x axis
[X, Y, Z] = meshgrid(xRange,yRange,zRange); % all combinations of x, y, z
lo = 0.09;hi=.20;
s=['x','o','*','s','+','d','v'];
c=['r','g','k','c','b'];
for i=1:C
    W = mvnpdf([X(:) Y(:) Z(:)], means(i,:), sigmas(:,:,i)); % compute pdf
    W = reshape(W/max(W),size(X));
    sc=scatter3(X((lo<W) & (W<hi)), Y((lo<W) & (W<hi)), Z((lo<W) & (W<hi)));
    sc.Marker=s(i);
    sc.MarkerEdgeColor=c(i);
end
title(['Actual data and mixtures, MDL = ',  num2str(MDL_true)]);

% Run EM algorithm for comparison
rng(SEED); % reset random seed to ensure reproducibility
disp('Running EM alone');
tic
Pop = InitPopulation(data, 15, C); % initialize (requires same M and K >= C to match GA-EM init)
newP = Pop(C); % only use the C^th one, which has correct components enabled
EM_result = EM(newP, data, Inf);
MDL_EM = MDLencode(EM_result,data); % compute MDL value for EM result
EMtime = toc;
disp(['  Time: ', num2str(EMtime), ' sec']);
disp(['EM: ', num2str(MDL_EM)]); % print MDL value

% plot EM-mixture
subplot(2,2,2);
scatter3(data(:,1), data(:,2), data(:,3),'filled');
hold on;
for i=1:C
    W = mvnpdf([X(:) Y(:) Z(:)], EM_result.means(:,i)', EM_result.covs(:,:,i)); % compute Gaussian pdf
    W = reshape(W/max(W),size(X));
    lo = 0.09;hi=.20;
    sc=scatter3(X((lo<W) & (W<hi)), Y((lo<W) & (W<hi)), Z((lo<W) & (W<hi)));
    sc.Marker=s(i);
    sc.MarkerEdgeColor=c(i);
end
title(['Best EM Mixture (', num2str(C), ' clusters), MDL = ',  num2str(MDL_EM)]);

% Run GA_EM algorithm (plots of fits and MDL vs iteration are generated
% internally)
rng(SEED); % reset random seed
disp('Running GA-EM');
tic
[GA_EM_result, MDL_list] = GA_EM(data, R, M, K, H, p_m);
GA_EMtime = toc;
disp(['  Time: ', num2str(GA_EMtime), ' sec']);
disp(['GA_EM: ', num2str(MDL_list(end))]); % print final MDL value

% plot GA_EM-mixture
subplot(2,2,3);
scatter3(data(:,1), data(:,2), data(:,3),'filled');
hold on;
index = 1:M;
index = index(logical(GA_EM_result(1).code));
for i=1:length(index)
    W = mvnpdf([X(:) Y(:) Z(:)], GA_EM_result.means(:,index(i))', GA_EM_result.covs(:,:,index(i))); % compute Gaussian pdf
    W = reshape(W/max(W),size(X));
    lo = 0.09;hi=.20;
    sc=scatter3(X((lo<W) & (W<hi)), Y((lo<W) & (W<hi)), Z((lo<W) & (W<hi)));
    sc.Marker=s(i);
    sc.MarkerEdgeColor=c(i);
end
title(['Best GA-EM Mixture (', num2str(sum(GA_EM_result(1).code)), ' clusters), MDL = ',  num2str(MDL_list(end))]);

% show MDL plot
subplot(2,2,4);
hold off;
plot(1:length(MDL_list), MDL_list(1:length(MDL_list)))
title(['MDL vs iterations (', num2str(length(MDL_list)), ')']);

% Output ratio of execution times
disp(['Ratio of execution times: ', num2str(GA_EMtime/EMtime)]);
% end
