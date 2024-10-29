source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

modelName = 'lambda121v1'
dataPath = '../../'
outputDir = 'export/'


#### Autopilot from here:

modelPath = pathConcat(dataPath, '/model/')
modelLogFile = paste0(modelPath, modelName, '.log')
modelJsonFile = paste0(modelPath, modelName, '.json')


if (!dir.exists(pathConcat(dataPath, outputDir))) {
  dir.create(pathConcat(dataPath, outputDir), recursive = T)
}


log = read.csv(modelLogFile, header = T)

{
  png(paste0(pathConcat(dataPath, outputDir), modelName, ' Accuracy and Loss.png'),
      width = 1200, height = 800, res = 100)
  
  par(mfrow = c(2,1), plt = c(0.15, 0.95, 0.25, 0.95))
  plot(log$epoch,
       log$accuracy,
       ylim = c(0.85, 1),
       type = 'l',
       lwd = 2,
       xaxs = 'i',
       yaxs = 'i',
       ylab = 'Accuracy (%)',
       xlab = 'Epoch',
       xlim = range(pretty(log$epoch)))
  lines(log$epoch, log$val_accuracy, lwd = 2, col = 'blue')
  lines(runmed(log$epoch,15), runmed(log$val_accuracy, 15), lwd = 2, col = 'black')
  grid()
  
  plot(log$epoch,
       log$loss,
       ylim = c(0, 1),
       type = 'l',
       lwd = 2,
       xaxs = 'i',
       yaxs = 'i',
       ylab = 'Loss',
       xlab = 'Epoch',
       xlim = range(pretty(log$epoch)))
  lines(log$epoch, log$val_loss, lwd = 2, col = 'blue')
  lines(runmed(log$epoch,15), runmed(log$val_loss, 15), lwd = 2, col = 'black')
  grid()
  
  dev.off()
}


sidecar = jsonlite::read_json(modelJsonFile)
trainingSet = list.files(sidecar$config$training$scnn_dir, pattern = '.png', full.names = T, recursive = T)
trainingSet = gsub(sidecar$config$training$scnn_dir, '', trainingSet)


