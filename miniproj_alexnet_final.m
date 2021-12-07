
%Form Dataset
imds = imageDatastore('data', 'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
numClasses = numel(categories(imdsTrain.Labels));

%import net and find its input size
net = alexnet_primary;
%load baseline_model;
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
layers = [ layersTransfer;
            fullyConnectedLayer(numClasses, 'WeightLearnRateFactor',1,'BiasLearnRateFactor',1);
            softmaxLayer;
            classificationLayer];

%Set parameters for data augmentation and resizing       
pixelRange = [-20 20];
rotateRange = [-2 2];
scaleRange = [.5 1];
imageAugmenter = imageDataAugmenter(...
                   'RandXReflection', true,...
                   'RandScale', scaleRange, ...
                   'RandXTranslation', pixelRange,...
                   'RandRotation', rotateRange, ...
                   'RandYTranslation', pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
                                       'ColorPreprocessing','gray2rgb',...
                                       'DataAugmentation', imageAugmenter);
% 
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation,...
                                            'ColorPreprocessing','gray2rgb');

% %set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'ValidationPatience',20, ...
    'Verbose',true, ...
    'Plots','training-progress');

%train net
netTransfer = trainNetwork(augimdsTrain,layers,options);
% 
% %test net
[YPred,scores] = classify(netTransfer,augimdsValidation);
% 
 YValidation = imdsValidation.Labels;
 accuracy = mean(YPred == YValidation);
% % 
idx = randperm(numel(imdsValidation.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = strcat('Pred: ',cellstr(YPred(idx(i))),' Actual: ',cellstr(YValidation(idx(i))));
    title(string(label));
end

alexnet_final = netTransfer;