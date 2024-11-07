function [trainData, valData, binaryLabels, imageNames] = loadImageData(imageDir, labelDir, inputSize,numClasses)
    miniBatchSize = 32;
    imageFiles = dir(fullfile(imageDir, '*.jpg'));
    
    % create the imageMap
    % get names and locations of image
    imageMap = containers.Map();
    for i = 1:numel(imageFiles)
        [~, name, ~] = fileparts(imageFiles(i).name); 
        imageMap(name) = fullfile(imageFiles(i).folder, imageFiles(i).name);
    end

    % get all the labels infos
    labelFiles = dir(fullfile(labelDir, '*.cls'));
    labelPaths = fullfile({labelFiles.folder}, {labelFiles.name})';

    numSamples = numel(labelFiles);
    
    % Pre-allocate arrays
    binaryLabels = zeros(numSamples, numClasses);
    imageNames = cell(numSamples, 1);
    imgFiles = strings(numSamples, 1);

    for i = 1:numSamples
        [~, name, ~] = fileparts(labelFiles(i).name);

        % check the exisetense
        if ~isKey(imageMap, name)
            error('Can not find the label %s who is linked to image。', labelFiles(i).name);
        end

        imageNames{i} = name;
        imgFiles(i)= imageMap(name);
    
        % 
        lblPath = labelPaths{i};
        lbl = readLabelCLS(lblPath); 
        lblIndices = lbl + 1; 
        binaryLabels(i, lblIndices) = 1;
    end

    
    dataTable = table(Size=[numSamples 2], ...
    VariableTypes=["string" "double"], ...
    VariableNames=["File_Location" "Labels"]);

    dataTable.File_Location = imgFiles;
    dataTable.Labels = binaryLabels;

    rng(0); 
    numTrain = round(0.8 * numSamples);
    indices = randperm(numSamples);
    trainIndices = indices(1:numTrain);
    valIndices = indices(numTrain+1:end);

    trainTable = dataTable(trainIndices, :);
    valTable = dataTable(valIndices, :);

    trainData = augmentedImageDatastore(inputSize(1:2), trainTable, ...
        'ColorPreprocessing', 'gray2rgb');
    trainData.MiniBatchSize = miniBatchSize;

    valData = augmentedImageDatastore(inputSize(1:2), valTable, ...
        'ColorPreprocessing', 'gray2rgb');
    valData.MiniBatchSize = miniBatchSize;
end

% read label data from cls
function labels = readLabelCLS(filename)

    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open: %s', filename);
    end
    lines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);

    allText = strjoin(lines{1}, ' ');

    tokens = regexp(allText, '[\d]+', 'match');
    labels = str2double(tokens);
end

trainImageDir = './images/train-resized';   
trainLabelDir = './labels/train';            
numClasses = 80;                             
inputSize = [224, 224, 3];

[trainData, valData, binaryLabels, imageNames] = loadImageData(trainImageDir, trainLabelDir,inputSize,numClasses);
