
%net = imagePretrainedNetwork("resnet50",NumClasses=numClasses);
net = imagePretrainedNetwork("resnet18",NumClasses=numClasses);

% layers = [
%     imageInputLayer(inputSize, 'Name', 'input')
% 
%     % layer1
%     convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
%     batchNormalizationLayer('Name', 'bn1')
%     reluLayer('Name', 'relu1')
%     maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
% 
%     % layer 2
%     convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
%     batchNormalizationLayer('Name', 'bn2')
%     reluLayer('Name', 'relu2')
%     maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
% 
%     % layer FC
%     fullyConnectedLayer(256, 'Name', 'fc1')
%     reluLayer('Name', 'relu3')
%     dropoutLayer(0.5, 'Name', 'dropout')
% 
%     fullyConnectedLayer(numClasses, 'Name', 'fc2')
%     sigmoidLayer('Name', 'sigmoid') 
% ];

options = trainingOptions("sgdm", ...
    InitialLearnRate=0.01, ...
    MiniBatchSize=32, ...
    MaxEpochs= 1, ...
    Verbose= false, ...
    ValidationData=valData, ...
    ValidationFrequency=100, ...
    ValidationPatience=5, ...
    Metrics="accuracy", ...
    Plots="training-progress");

trainedNet = trainnet(trainData,net,"binary-crossentropy",options);