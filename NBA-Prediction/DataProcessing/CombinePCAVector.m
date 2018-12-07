clear all;
a = load('../Data/PCAVector/all_pca_vector_2010-2017.mat');
b = load('../Data/PCAVector/2017FeaturesPCA.mat');
FeaturesLabel = [a.FeaturesLabel ; b.FeaturesLabel];
save('../Data/PCAVector/all_pca_vector_2010-2018.mat', 'FeaturesLabel')

