close all;

layer = sigmoidLayer('Name', 'sig1');

%net = imagePretrainedNetwork("mobilenetv2",NumClasses=numClasses);
%net = replaceLayer(net,"Logits_softmax",layer);

%net = imagePretrainedNetwork("resnet18",NumClasses=numClasses);
%net = replaceLayer(net,"prob",layer);

%net = imagePretrainedNetwork("resnet50",NumClasses=numClasses);
%net = replaceLayer(net,"fc1000_softmax",layer);

net = imagePretrainedNetwork("resnet101",NumClasses=numClasses);
net = replaceLayer(net,"prob",layer);

%analyzeNetwork(net);

%net = freezeNetwork(net,LayerNamesToIgnore="Logits"); % pour mobilenetv2
net = freezeNetwork(net,LayerNamesToIgnore="fc1000"); % pour resnet18/resnet50/resnet101

miniBatchSize = 64;
miniBatchPerEpoch = floor(numTrain/miniBatchSize);
valFreq = floor(miniBatchPerEpoch * 0.5);

accuracy = accuracyMetric(ClassificationMode="multilabel", AverageType="macro");
precision = precisionMetric(ClassificationMode="multilabel", AverageType="macro");
fscore = fScoreMetric(ClassificationMode="multilabel", AverageType="macro");
metric = {fscore};

options = trainingOptions("adam", ...
    InitialLearnRate=0.0005, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=5, ...
    L2Regularization=0.0001,...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs= 20, ...
    Verbose= false, ...
    ValidationData=valData, ...
    ValidationFrequency=valFreq, ...
    ValidationPatience=10, ...
    Metrics=metric, ...
    Plots="training-progress");

trainedNet = trainnet(trainData,net,"binary-crossentropy",options);