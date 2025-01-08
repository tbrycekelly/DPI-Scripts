library(jsonlite)
source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

modelPath = '../../model/'

#### Autopilot from here:


modelLogFiles = list.files(modelPath, pattern = '.log')
modelJsonFiles = list.files(modelPath, pattern = '.json')

sidecar = list()

for (i in 1:length(modelJsonFiles)) {
  tmpName = gsub('.json', '', modelJsonFiles[i])
  ## Load logs
  sidecar[[tmpName]] = jsonlite::read_json(paste0(modelPath, modelJsonFiles[i]))
  sidecar[[tmpName]]$log = read.csv(paste0(modelPath, gsub('.json', '.log', modelJsonFiles[i])))
}

{
  plot(NULL, NULL,
       xlim = c(3, 6),
       ylim = c(0,1),
       xaxt = 'n',
       yaxt = 'n',
       xaxs = 'i',
       yaxs = 'i',
       xlab = 'Training Time (s)',
       ylab = 'Validation Loss')
  axis(1, at = c(0:10), labels = 10^c(0:10))
  for (j in 0:10) {
    axis(1, at = log10(1:9) + j, labels = NA, col.ticks = 'grey')
  }
  abline(v = log10(c(60, 3600, 86400)), lty = 3, col = 'grey')
  
  axis(2, at = seq(0, 3, by = 0.1), labels = NA, col.ticks = 'grey')
  axis(2, at = c(0:3), las = 1)
  abline(h = seq(0, 3, by = 0.1), lty = 3, col = 'grey')
  
  for (i in 1:length(sidecar)) {
    durration = sidecar[[i]]$timings$model_train_end - sidecar[[i]]$timings$model_train_start # seconds
    epochCount = max(sidecar[[i]]$log$epoch) + 1
    finalEpochs = epochCount + 1 - c(1:(epochCount/20))
    dt = (sidecar[[i]]$timings$model_train_end - sidecar[[i]]$timings$model_train_start) / sidecar[[i]]$config$training$stop
    
    points(log10(durration), sidecar[[i]]$log$val_loss[epochCount])
    lines(rep(log10(durration), 2), range(sidecar[[i]]$log$val_loss[finalEpochs]))
    
    text(log10(durration), sidecar[[i]]$log$val_loss[epochCount], labels = names(sidecar)[i], pos = 2, cex = 0.75)
  }
}



{
  plot(NULL, NULL,
       xlim = c(3, 6),
       ylim = c(0.9,1),
       xaxt = 'n',
       yaxt = 'n',
       xaxs = 'i',
       yaxs = 'i',
       xlab = 'Training Time (s)',
       ylab = 'Validation Loss')
  axis(1, at = c(0:10), labels = 10^c(0:10))
  for (j in 0:10) {
    axis(1, at = log10(1:9) + j, labels = NA, col.ticks = 'grey')
  }
  abline(v = log10(c(60, 3600, 86400)), lty = 3, col = 'grey')
  
  axis(2, at = seq(0, 3, by = 0.1), labels = NA, col.ticks = 'grey')
  axis(2, at = c(0:3), las = 1)
  abline(h = seq(0, 3, by = 0.1), lty = 3, col = 'grey')
  
  for (i in 1:length(sidecar)) {
    durration = sidecar[[i]]$timings$model_train_end - sidecar[[i]]$timings$model_train_start # seconds
    epochCount = max(sidecar[[i]]$log$epoch) + 1
    finalEpochs = epochCount + 1 - c(1:(epochCount/20))
    dt = (sidecar[[i]]$timings$model_train_end - sidecar[[i]]$timings$model_train_start) / sidecar[[i]]$config$training$stop
    
    points(log10(durration), sidecar[[i]]$log$val_accuracy[epochCount])
    lines(rep(log10(durration), 2), range(sidecar[[i]]$log$val_accuracy[finalEpochs]))
    
    text(log10(durration), sidecar[[i]]$log$val_accuracy[epochCount], labels = names(sidecar)[i], pos = 2, cex = 0.75)
  }
}

