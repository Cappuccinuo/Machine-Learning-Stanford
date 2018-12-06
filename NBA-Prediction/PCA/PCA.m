clear all;
load('AllFeatureVectors2.mat')

CompleteFeaturesGames = [0 0 0 0 0 0 0 0 0];
for i=4:4
[a,b] = find( FeatureVector{i}(:,:) == -9999);
minGame(i) = max(a); % find the last game that dont have all informations
% take just the games with all features
CompleteFeaturesGames = vertcat(CompleteFeaturesGames,FeatureVector{i}(minGame(i)+1:end,5:end));
end
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

save('2017FeaturesPCA','FeaturesLabel');




