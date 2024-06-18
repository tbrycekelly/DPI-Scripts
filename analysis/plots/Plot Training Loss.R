library(jsonlite)
source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

modelName = 'iota121v1'
dataPath = '../../'
outputDir = 'export/'
camera = 'camera0'
transect = 'test1'
segmentationName = 'REG'

#### Autopilot from here:

sourceFiles = list(
  rawPath = pathConcat(dataPath, '/raw/camera0/', transect),
  segmentationPath = pathConcat(dataPath, '/analysis/', camera, '/segmentation/', transect, '-', segmentationName),
  classificationPath = pathConcat(dataPath, '/analysis/', camera, '/classification/', transect, '-', segmentationName, '-', modelName),
  modelPath = pathConcat(dataPath, '/model/')
)

sourceFiles$segmentationFiles = list.files(sourceFiles$segmentationPath, pattern = 'statistics.csv')
sourceFiles$classificationFiles = list.files(sourceFiles$classificationPath, pattern = 'prediction.csv')
sourceFiles$rawFiles = list.files(sourceFiles$rawPath, pattern = '.avi')

if (!dir.exists(pathConcat(dataPath, outputDir))) {
  dir.create(pathConcat(dataPath, outputDir), recursive = T)
}

modelLogFile = paste0(sourceFiles$modelPath, modelName, '.log')
modelJsonFile = paste0(sourceFiles$modelPath, modelName, '.json')
modelValidationFile = paste0(sourceFiles$modelPath, modelName, ' predictions.csv')

sidecar = jsonlite::read_json(modelJsonFile)

predictions = read.csv(modelValidationFile, header = T)
colnames(predictions)[1] = 'true'
predictions$true = sidecar$labels[predictions$true+1]

p = rep(NA, nrow(predictions))

for (i in 1:length(p)) {
  p[i] = predictions[i,which(predictions$true[i] == colnames(predictions))]
}

loss = -log2(p)
hist(loss)
summary(loss)
mean(loss)
