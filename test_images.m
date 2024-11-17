close all;

testImageFolder = './images/test-resized2'; 

categoriesTrain = (0:79);

% read the test-images
imageFiles = dir(fullfile(testImageFolder, '*.jpg')); 
imageNames = string(fullfile(testImageFolder, {imageFiles.name}));

thresholdValue = 0.5;

figure
tiledlayout(1,length(imageFiles))
results = containers.Map();

for i = 1:length(imageFiles)
    img = imread(imageNames(i));
    img = imresize(img,inputSize(1:2));
    if isscalar(img(1,1,:))
        img = cat(3, img, img, img);
    end

    scoresImg = predict(trainedNet,single(img))';
    % disp(scoresImg)
    YPred =  categoriesTrain(scoresImg >= thresholdValue) ;
    if isscalar(YPred)
        YPred = {YPred};
    end

    [~, imageName, ~] = fileparts(imageFiles(i).name);
    results(imageName) = YPred;

    nexttile
    imshow(img)
    title(YPred)

end

jsonData = jsonencode((results),PrettyPrint=true);


% save the result
jsonFileName = 'predicted_labels1.json';
fid = fopen(jsonFileName, 'w');
if fid == -1
    error('Can not create');
end
fwrite(fid, jsonData, 'char');
fclose(fid);

disp(['Already save', jsonFileName]);