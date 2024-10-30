source('processing/low level utilities.R')
source('processing/mid level utilities.R')
library(furrr)

#### User input: 

camera = 'camera2'
transect = 'SKQJ22'
segmentationName = 'REG'
modelName = 'kappa121v20'
dataPath = '../../'
outputDir = '../../export/classification/'
pMin = 0.5


sidecar = list(
  camera = 'camera2',
  transect = 'SKQJ22',
  segmentationName = 'REG',
  modelName = 'kappa121v20',
  dataPath = '../../',
  outputDir = '../../export/classification/',
  pMin = 0.5,
  timings = list(),
  system = list(
    Sys.info(),
    Sys.getenv()
  )
)

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

message('Found ', length(sourceFiles$classificationFiles), ' classification results.')

## Calculate priors from full dataset:
nSample = max(min(500, length(sourceFiles$classificationFiles)), length(sourceFiles$classificationFiles) * 0.1)
l = sample(1:length(sourceFiles$classificationFiles), size = nSample)

classification = read.csv(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[l[1]]), header = T)
for (i in 2:nSample) {
  message(round(i / nSample * 100), '%')
  tmp = read.csv(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[l[i]]), header = T)
  classification = rbind(classification, tmp)
}

prior = getPrior(classification[,-1])

weight = matrix(prior$weight, nrow = 1, ncol = length(prior$weight), byrow = T)
prior = getPrior(classification[,-1] * weight)

weight = matrix(prior$weight, nrow = 1, ncol = length(prior$weight), byrow = T)
prior = getPrior(classification[,-1] * weight)

weight = matrix(prior$weight, nrow = 1, ncol = length(prior$weight), byrow = T)
prior = getPrior(classification[,-1] * weight)

weight = matrix(prior$weight, nrow = 1, ncol = length(prior$weight), byrow = T)
prior = getPrior(classification[,-1] * weight)

weight = matrix(prior$weight, nrow = 1, ncol = length(prior$weight), byrow = T)
prior[order(prior$n),]

if (!dir.exists(outputDir)) {
  dir.create(paste0(outputDir, '/', 'preprocessed/'), recursive = T, showWarnings = F)
}
rm(classification)

startTime = Sys.time()
plan(multisession)
future_map(sourceFiles$classificationFiles, preprocessPrediction)
plan(sequential)
endTime = Sys.time()

sidecar$timings$preprocessing = endTime - startTime



#### Load and process each classification file:
preprocessPrediction = function(classificationFile) {
  
  predictions = read.csv(paste0(sourceFiles$classificationPath, classificationFile), header = T)
  classes = colnames(predictions)[-1]
  frameNum = as.numeric(sapply(predictions$X, function(x) {as.numeric(strsplit(x, '-')[[1]][8])}))
  
  ## Determine when:
  tmp = strsplit(classificationFile, ' ')[[1]]
  tmpDate = strsplit(tmp, '-')
  datetime = as.POSIXct(paste0(tmpDate[[1]][3], '-', tmpDate[[1]][4], '-', tmpDate[[1]][5], ' ',
                               tmpDate[[2]][1], ':', tmpDate[[2]][2], ':', substr(tmpDate[[2]][3], 1, 2)), tz = 'America/Anchorage')
  datetime = as.numeric(datetime) + frameNum / 20.52
  datetime = round(datetime/5)*5 # TODO
  
  # Calculate the resulting probability and the prediction index:
  pmax = apply(predictions[,-1] * weight, 1, function(x){max(x) / sum(x)})
  pIndex = apply(predictions[,-1] * weight, 1, which.max) # Weighted based on prior
  
  l = pmax >= pMin
  tmp = data.frame(datetime = datetime[l], index = pIndex[l])
  tmp = tmp[!is.na(tmp$datetime),]
  
  predicted = data.frame(datetime = Sys.time())
  for (n in prior$class) {
    predicted[[n]] = NA
  }
  
  for (d in unique(tmp$datetime)) {
    k = which(tmp$datetime == d)
    counts = sapply(1:length(weight), function(x){sum(x == tmp$index[k])})
    predicted = rbind(predicted, c(d, counts))
  }
  saveRDS(file = paste0(outputDir, '/', 'preprocessed/', gsub('.csv', '.rds', classificationFile)),
          predicted[-1,])
}


preprocessedPredictions = list.files(paste0(outputDir, '/', 'preprocessed/'), pattern = '.rds', full.names = T)

prediction = readRDS(preprocessedPredictions[1])
for (i in 2:length(preprocessedPredictions)) {
  message(i)
  tmp = readRDS(preprocessedPredictions[i])
  prediction = rbind(prediction, tmp)
}

prediction = prediction[order(prediction$datetime),]
prediction = prediction[!is.na(prediction$datetime),]

cbind(colnames(prediction), 1:ncol(prediction))

prediction$copepod = apply(prediction[,c(9:13, 17:21)],1, sum)
prediction$ctenophore = apply(prediction[,c(25:32)],1, sum)

nominalVol = 3.15 * 20.52 * 5 * 1e-3 # m-3

hist(log10(prediction$copepod / nominalVol + 0.1), pch = '.', breaks = 20)
hist(log10(prediction$ctenophore / nominalVol + 0.1), pch = '.', breaks = 150)
plot(prediction$ctenophore, pch = '.')

## Matchup Depth, Lat, Lon, T, S, Cast to prediction:
sensor = readRDS('../../sensor/SKQ202210S DPI Interpolated.rds')
sensor$time = as.numeric(sensor$time)

prediction$latitude = approx(sensor$time, sensor$latitude, xout = prediction$datetime)$y
prediction$longitude = approx(sensor$time, sensor$longitude, xout = prediction$datetime)$y
prediction$depth = approx(sensor$time, sensor$depth, xout = prediction$datetime)$y







