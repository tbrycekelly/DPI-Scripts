source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

outputDir = '../../export/prior1'
nMax = 1000 # Maximum number of images per pull folder

camera = 'camera1'
transect = 'SewardLine_Summer22'
segmentationName = 'REG'
modelName = 'iota201v1'
dataPath = '../../'
outputDir = 'export/classification/'


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


## Determine the predicted distribution of particle types:

classification = loadPrediction(paste0(sourceFiles$classificationPath, '/', sourceFiles$classificationFiles[1]))

if (length(sourceFiles$classificationFiles) > 1) {
  for (i in 2:length(sourceFiles$classificationFiles)) {
    tmp = loadPrediction(paste0(sourceFiles$classificationPath, '/', sourceFiles$classificationFiles[1]))
    classification = rbind(classification, tmp)
  }
}

getPrior = function(classification) {
  prediction = apply(classification, 1, which.max)
  stats = data.frame(class = colnames(classification), n = NA, weight = NA)
  for (i in 1:nrow(stats)) {
    k = which(prediction == i)
    stats$n[i] = length(k)
  }
  stats$weight = stats$n / nrow(classification) + 0.5 / nrow(stats)
  stats$weight = stats$weight / sum(stats$weight)
  stats
}


#prediction = colnames(classification)[-c(1:4)][apply(classification[,-c(1:4)], 1, which.max)]
prior = getPrior(classification[,-c(1:4)])

weight = matrix(prior$weight, nrow = nrow(classification), ncol = length(prior$weight), byrow = T)
prior2 = getPrior(classification[,-c(1:4)] * weight)

weight = matrix(prior$weight2, nrow = nrow(classification), ncol = length(prior$weight2), byrow = T)
prior3 = getPrior(classification[,-c(1:4)] * weight)

weight = matrix(prior$weight3, nrow = nrow(classification), ncol = length(prior$weight3), byrow = T)
prior4 = getPrior(classification[,-c(1:4)] * weight)

weight = matrix(prior$weight4, nrow = nrow(classification), ncol = length(prior$weight4), byrow = T)
prior5 = getPrior(classification[,-c(1:4)] * weight)

data.frame(class = prior$class,
           n1 = prior$n,
           n2 = prior2$n,
           n3 = prior3$n,
           n4 = prior4$n,
           n5 = prior5$n)


## PULL different images
if (!dir.exists(outputDir)) {
  dir.create(outputDir, recursive = T)
}

initTime = Sys.time()
for (i in 1:length(sourceFiles$classificationFiles)) {
  startTime = Sys.time()
  
  predictions = read.csv(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[i]), header = T)
  
  ## Extract ROIs
  zipName = paste0(sourceFiles$segmentationPath, gsub('_prediction.csv', '.zip', sourceFiles$classificationFiles[i]))
  extractName = paste0(sourceFiles$segmentationPath, gsub('_prediction.csv', '/', sourceFiles$classificationFiles[i]))
  unzip(zipfile = zipName, exdir = extractName)
  
  weight = matrix(prior5$weight, nrow = nrow(predictions), ncol = length(prior5$weight), byrow = T)
  
  classes = colnames(predictions)[-1]
  pmax = apply(predictions[,-1], 1, max)
  pmax = round(pmax, digits = 2)
  pIndex = apply(predictions[,-1], 1, which.max)
  pIndexPrior = apply(predictions[,-1] * weight, 1, which.max)
  
  for (class in classes) {
    if (!dir.exists(paste0(outputDir, '/', class))) {
      dir.create(paste0(outputDir, '/', class), recursive = T)
    }
  }
  
  l = pIndex != pIndexPrior
  
  file.copy(from = paste0(extractName, '/', predictions$X)[l], 
            to = paste0(outputDir, '/', classes[pIndexPrior], '/', classes[pIndex], ' to ', classes[pIndexPrior], '-', gsub('.png', '', predictions$X), '.png')[l])
  unlink(extractName, recursive = T)
  endTime = Sys.time()
  
  message('Finished file ', i, ' out of ', length(sourceFiles$classificationFiles),
          '.\tElapsed time: ', round(as.numeric(endTime) - as.numeric(initTime)), ' sec',
          '\tRemaining time:', round((as.numeric(endTime) - as.numeric(initTime)) / i * (length(sourceFiles$classificationFiles) - i) / 60), ' min')
}

initTime = Sys.time()
for (i in 1:length(sourceFiles$classificationFiles)) {
  startTime = Sys.time()
  
  predictions = read.csv(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[i]), header = T)
  
  ## Extract ROIs
  zipName = paste0(sourceFiles$segmentationPath, gsub('_prediction.csv', '.zip', sourceFiles$classificationFiles[i]))
  extractName = paste0(sourceFiles$segmentationPath, gsub('_prediction.csv', '/', sourceFiles$classificationFiles[i]))
  unzip(zipfile = zipName, exdir = extractName)
  
  weight = matrix(prior5$weight, nrow = nrow(predictions), ncol = length(prior5$weight), byrow = T)
  
  classes = colnames(predictions)[-1]
  pmax = apply(predictions[,-1], 1, max)
  pmax = round(pmax, digits = 2)
  pIndex = apply(predictions[,-1], 1, which.max)
  pIndexPrior = apply(predictions[,-1] * weight, 1, which.max)
  
  for (class in classes) {
    if (!dir.exists(paste0(outputDir, '/', class))) {
      dir.create(paste0(outputDir, '/', class), recursive = T)
    }
  }
  
  l = pIndex != pIndexPrior
  
  file.copy(from = paste0(extractName, '/', predictions$X)[l], 
            to = paste0(outputDir, '/', classes[pIndex], '/', classes[pIndex], ' to ', classes[pIndexPrior], '-', gsub('.png', '', predictions$X), '.png')[l])
  unlink(extractName, recursive = T)
  endTime = Sys.time()
  
  message('Finished file ', i, ' out of ', length(sourceFiles$classificationFiles),
          '.\tElapsed time: ', round(as.numeric(endTime) - as.numeric(initTime)), ' sec',
          '\tRemaining time:', round((as.numeric(endTime) - as.numeric(initTime)) / i * (length(sourceFiles$classificationFiles) - i) / 60), ' min')
}


## Apply approach to training set predictions.

trainingPredictions = read.csv('../../model/iota201v2 predictions.csv')
pred1 = apply(trainingPredictions[,-1], 1, which.max)

prior = getPrior(trainingPredictions[,-1])
weights = matrix(prior$weight, nrow = nrow(trainingPredictions), ncol = ncol(trainingPredictions)-1, byrow = T)
pred2 = apply(trainingPredictions[,-1] * weights, 1, which.max)

prior = getPrior(trainingPredictions[,-1] * weights)

performance = data.frame(true = trainingPredictions$X+1,
                         normalPrediction = pred1,
                         priorPrediction = pred2)

sum(performance$true != performance$normalPrediction)
sum(performance$true != performance$priorPrediction)
sum(performance$normalPrediction != performance$priorPrediction)

performance[which(performance$normalPrediction != performance$priorPrediction),]
