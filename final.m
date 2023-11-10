mainFolder = 'project_image/';
outputFolder = 'hog_features/';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

subfolders = dir(mainFolder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));

features = [];
labels = {};
cellSize = [8, 8];
blockSize = [2, 2];
numBins = 9;

for i = 1:length(subfolders)
    subfolderName = subfolders(i).name;
    subfolderPath = fullfile(mainFolder, subfolderName);
    subfolderOutput = fullfile(outputFolder, subfolderName);

    if ~exist(subfolderOutput, 'dir')
        mkdir(subfolderOutput);
    end
    files = dir(fullfile(subfolderPath, '*.jpg'));

    for j = 1:length(files)
        imagePath = fullfile(subfolderPath, files(j).name);
        image = imread(imagePath);
        image = rgb2gray(image);
        image = imresize(image, [1000, 800]);
        featuresVector = extractHOGFeatures(image, 'CellSize', cellSize, 'BlockSize', blockSize, 'NumBins', numBins);
        [~, baseFileName, ~] = fileparts(files(j).name);
        savePath = fullfile(subfolderOutput, [baseFileName, '_hog_features.mat']);
        save(savePath, 'featuresVector');
        features = [features; featuresVector];
        labels = [labels; subfolderName];
    end

end

k = 3;

knnModel = fitcknn(features, labels, 'NumNeighbors', k);
newImage = imread('DEMOS/Angry demo.jpg');
imshow(newImage);
newImage = rgb2gray(newImage);
newImage = imresize(newImage, [1000, 800]);

newFeatures = extractHOGFeatures(newImage, 'CellSize', cellSize, 'BlockSize', blockSize, 'NumBins', numBins);
predictedLabel = predict(knnModel, newFeatures);
disp(['Predicted Mood: ', predictedLabel]);
