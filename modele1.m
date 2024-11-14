close all;

layer = sigmoidLayer('Name', 'sig1');

net = imagePretrainedNetwork("mobilenetv2",NumClasses=numClasses);
net = replaceLayer(net,"Logits_softmax",layer);

%net = imagePretrainedNetwork("resnet18",NumClasses=numClasses);
%net = replaceLayer(net,"prob",layer);

%net = imagePretrainedNetwork("resnet50",NumClasses=numClasses);
%net = replaceLayer(net,"fc1000_softmax",layer);

%net = imagePretrainedNetwork("resnet101",NumClasses=numClasses);
%net = replaceLayer(net,"prob",layer);

%analyzeNetwork(net);

%net = freezeNetwork(net,LayerNamesToIgnore="Logits"); % pour mobilenetv2
%net = freezeNetwork(net,LayerNamesToIgnore="fc1000"); % pour resnet18/resnet50/resnet101

miniBatchSize = 64;
miniBatchPerEpoch = floor(numTrain/miniBatchSize);
valFreq = floor(miniBatchPerEpoch * 0.5);

options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs= 10, ...
    Verbose= false, ...
    ValidationData=valData, ...
    ValidationFrequency=valFreq, ...
    ValidationPatience=5, ...
    Metrics="accuracy", ...
    Plots="training-progress");

%trainedNet = trainnet(trainData,net,"binary-crossentropy",options);
disp(valFreq)