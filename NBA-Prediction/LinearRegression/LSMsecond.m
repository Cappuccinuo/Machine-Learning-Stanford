clear all;
load ('../Data/PCAVector/all_pca_vector_2010-2017.mat')
FeatureVectorsTr = FeaturesLabel;
w(1) = 0;
w(2) = 0;
w(3) = 0;
w(4) = 0;

stepSize = 0.00001;
% implement of the LSM algorithm
divisor = length(FeatureVectorsTr);
n=1;
for k=0:300000
    y(n) = sum(w(k+1,:).*[1 FeatureVectorsTr(n,1:3)]);
    e(n) = FeatureVectorsTr(n,4) - y(n);
    w(k+2,:) = w(k+1,:) + stepSize.*e(n).*[1 FeatureVectorsTr(n,1:3)];
    n=n+1;
    if n > divisor
        n = 1;
    end
end

wFinal = mean(w(size(w,1)*9/10:end,:));

for i=1:size(w,2)
figure
plot(w(1:10:end,i))
end