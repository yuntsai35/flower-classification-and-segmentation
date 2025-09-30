close all;
clear;

digitDatasetPath = fullfile(pwd, '17flowers');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

inputSize = [256 256 3];

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-2 2], ...
    'RandScale',[0.9 1.1], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);

augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augImdsVal = augmentedImageDatastore(inputSize, imdsVal);
augImdsTest = augmentedImageDatastore(inputSize, imdsTest);

numClasses = 17;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5, 32)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 64)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...     
    'MaxEpochs', 10, ...    
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', augImdsVal, ...
    'ValidationFrequency', 10);

net = trainNetwork(augImdsTrain, layers, options);
save('classnet.mat', 'net');

YPredVal = classify(net, augImdsVal);
YVal = imdsVal.Labels;

accuracyVal = mean(YPredVal == YVal);
disp(['Validation Accuracy：', num2str(accuracyVal*100), '%']);

YPredTest = classify(net, augImdsTest);
YTest = imdsTest.Labels;

accuracyTest = mean(YPredTest == YTest);
disp(['Test Accuracy：', num2str(accuracyTest*100), '%']);

figure;
confusionchart(YVal, YPredVal);
title('Confusion Matrix - Validation Set');

figure;
confusionchart(YTest, YPredTest);
title('Confusion Matrix - Test Set');
