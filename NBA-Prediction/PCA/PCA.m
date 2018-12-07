clear all;
%load('../Data/AllFeatureVector/All_Vectors.mat')
load('../Data/AllFeatureVector/2017-2018_Vectors.mat')
s = 2017;
e = 2018;
base = 2017;
for year=s:e - 1
    CompleteFeaturesGames = [0 0 0 0 0 0 0 0 0];
    j = year - base + 1;
    [a,b] = find( FeatureVector{j}(:,:) == -9999);
    minGame(j) = max(a); % find the last game that dont have all informations
    % take just the games with all features
    CompleteFeaturesGames = vertcat(CompleteFeaturesGames,FeatureVector{j}(minGame(j)+1:end,5:end));
    
    CompleteFeaturesGames = CompleteFeaturesGames(2:end,:);
    K = length(CompleteFeaturesGames); % number of feature vectors
    data = CompleteFeaturesGames(:,1:end-1); % exclude the label
    meanVector = sum(data)';
    % covariance matrix
    aux = zeros(8,8);
    for i=1:10
        aux = aux + ((data(i,:)')-meanVector)*(((data(i,:))'-meanVector)');
    end
    C = aux/K;

    % Compute Eigen Vectors and Values
    % U columns are eigenvectors and S has the eigenvalues
    [U,S,V] = svd(C);

    %Principal component analysis r=3
    r=3;
    Vp = U(:,1:r);
    Sp = S(1:r,1:r);
    % Project each feature vector x(k) onto the r-dimensional principle subspace
    Y = (Vp'*data')';
    FeaturesLabel = horzcat(Y,CompleteFeaturesGames(:,9));
    path = '../Data/PCAVector/';
    name = strcat(path, int2str(year), 'FeaturesPCA');
    save(name,'FeaturesLabel');
end





