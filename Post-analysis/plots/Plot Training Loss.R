library(jsonlite)
source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

modelName = 'resnet152-1'
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

## Load logs
sidecar = jsonlite::read_json(modelJsonFile)
logfile = read.csv(modelLogFile)
dt = (sidecar$timings$model_train_end - sidecar$timings$model_train_start) / max(logfile$epoch + 1) # seconds

plot(logfile$epoch,
     pmin(logfile$val_loss, 10),
     type = 'l',
     ylim = c(0, 10),
     yaxs = 'i',
     xlim = c(0, 400),
     xaxs = 'i',
     xlab = 'Epoch',
     ylab = 'Validation Loss',
     lwd = 3)
grid()

f = function(a, b, c, d, x) {
  a * exp(b*x + c) + d
}

expFit = function(x, y, xout = NULL) {
  if (is.null(xout)) {
    xout = x
  }
  # y = A * exp(x*B + C) + D
  initGuess = c(3, 0, 0, 0.3)
  score = sum((y - f(initGuess[1], initGuess[2], initGuess[3], initGuess[4], x))^2)
  
  for (i in 1:3e6) {
    newGuess = initGuess + rnorm(length(initGuess), mean = 0, sd = c(0.01, 0.01, 0.000001, 0.01))
    newScore = sum((y - f(newGuess[1], newGuess[2], newGuess[3], newGuess[4], x))^2)
    
    if (newScore - score < runif(1)) {
      score = newScore
      initGuess = newGuess
    }
  }
  
  message('Best fit has score = ', round(score, 3), ':\ta=', initGuess[1], '\tb=', initGuess[2], '\tc=', initGuess[3], '\td=', initGuess[4])
  
  f(initGuess[1], initGuess[2], initGuess[3], initGuess[4], xout)
}

fit = expFit(logfile$epoch, logfile$val_loss)
lines(logfile$epoch, fit, col = 'red')
