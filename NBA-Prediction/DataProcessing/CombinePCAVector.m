clear all;
a = load('../Data/PCAVector/2010FeaturesPCA.mat');
b = load('../Data/PCAVector/2011FeaturesPCA.mat');
c = load('../Data/PCAVector/2012FeaturesPCA.mat');
d = load('../Data/PCAVector/2013FeaturesPCA.mat');
e = load('../Data/PCAVector/2014FeaturesPCA.mat');
f = load('../Data/PCAVector/2015FeaturesPCA.mat');
g = load('../Data/PCAVector/2016FeaturesPCA.mat');
%b = load('../Data/PCAVector/2017FeaturesPCA.mat');
FeaturesLabel = [a.FeaturesLabel ; b.FeaturesLabel; c.FeaturesLabel; d.FeaturesLabel; e.FeaturesLabel; f.FeaturesLabel; g.FeaturesLabel];
save('../Data/PCAVector/all_pca_vector_2010-2017.mat', 'FeaturesLabel')

