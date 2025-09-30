close all;
clear;

dataDir = fullfile(pwd, 'daffodilSeg');
imDir = fullfile(dataDir, 'ImagesRsz256');
pxDir = fullfile(dataDir, 'LabelsRsz256');

imds = imageDatastore(imDir);
classNames = ["flowers", "background"];
pixelLabelID = [1, 3]; 
pxds = pixelLabelDatastore(pxDir, classNames, pixelLabelID);

C = readimage(pxds,1);
disp(C(5,5))
C = double(C);
C(C ~= 1) = 3; 
C = categorical(C, [1 3], classNames);

[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds, pxds);

inputSize = [256 256 3];
numClasses = 2;
unetNetwork = unetLayers(inputSize, numClasses);

trainingData = combine(imdsTrain, pxdsTrain);
validationData = combine(imdsVal, pxdsVal);

opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',5, ...
    'MiniBatchSize',64, ...
    'shuffle','every-epoch',...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(trainingData, unetNetwork, opts);

save('segmentnet.mat', 'net');

pxdsResults = semanticseg(imdsTest, net, 'WriteLocation', 'C:\temp_data');

overlayOut1 = labeloverlay(readimage(imdsTest,1), readimage(pxdsResults,1));
figure
imshow(overlayOut1);
title('Overlay of Image 1');

overlayOut2 = labeloverlay(readimage(imdsTest,2), readimage(pxdsResults,2));
figure
imshow(overlayOut2);
title('Overlay of Image 2');

metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTest, 'Verbose', false);

figure;
cm = metrics.ConfusionMatrix;  
chart=confusionchart(table2array(cm), cm.Properties.RowNames); 
chart.RowSummary = 'row-normalized';
chart.ColumnSummary = 'column-normalized';
title('Normalized Confusion Matrix'); 

meanIoU = metrics.DataSetMetrics.MeanIoU;
meanAccuracy = metrics.DataSetMetrics.MeanAccuracy;
fprintf('Mean IoU: %.4f\n', meanIoU);
fprintf('Overall Mean Accuracy: %.4f\n', meanAccuracy);

disp('Per-Class IoU and Accuracy:');
disp(table(metrics.ClassMetrics.Properties.RowNames, ...
           metrics.ClassMetrics.IoU, ...
           metrics.ClassMetrics.Accuracy, ...
    'VariableNames', {'ClassName', 'IoU', 'Accuracy'}));


function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds, pxds)
    rng(0); 
    numFiles = numel(imds.Files);
    shuffledIndices = randperm(numFiles);
    numTrain = round(0.6 * numFiles);
    numVal = round(0.2 * numFiles);
    trainingIdx = shuffledIndices(1:numTrain);
    valIdx = shuffledIndices(numTrain+1:numTrain+numVal);
    testIdx = shuffledIndices(numTrain+numVal+1:end);
    imdsTrain = subset(imds, trainingIdx);
    imdsVal = subset(imds, valIdx);
    imdsTest = subset(imds, testIdx);
    pxdsTrain = subset(pxds, trainingIdx);
    pxdsVal = subset(pxds, valIdx);
    pxdsTest = subset(pxds, testIdx);
end