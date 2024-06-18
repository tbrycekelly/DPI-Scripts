source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

camera = 'camera0'
transect = 'test1'
segmentationName = 'REG'
modelName = 'iota121v1'
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

if (!dir.exists(pathConcat(dataPath, outputDir))) {
  dir.create(pathConcat(dataPath, outputDir), recursive = T)
}


for (i in 1:length(sourceFiles$classificationFiles)) {
  predictions = read.csv(paste0(sourceFiles$classificationPath, '/', sourceFiles$classificationFiles[i]), header = T)
  class = apply(predictions[,-1], 1, which.max)
  
  counts = data.frame(category = colnames(predictions)[-1])
  counts$count = sapply(1:nrow(counts), function(j) { sum(class == j)})
  counts = counts[order(counts$count, decreasing = T),]
  
  {
    currentName = gsub('.avi_prediction.csv', '', sourceFiles$classificationFiles[i])
    png(filename = paste0(pathConcat(dataPath, outputDir), 'Class Distribution ', currentName, '.png'),
        width = 2000,
        height = 1600,
        res = 150)
    
    par(plt = c(0.12, 0.98, 0.3, 0.98))
    plot(NULL,
         NULL,
         xlim = c(1,nrow(counts)),
         ylim = c(0, 3.5),
         xlab = '',
         ylab = 'Counts',
         xaxt = 'n',
         yaxt = 'n')
    
    axis(1, at = c(1:nrow(counts)), labels = counts$category, las = 2, cex.axis = 0.7)
    for (j in seq(1, nrow(counts), by = 5)) {
      abline(v = j-0.5, col = 'grey')
    }
    
    abline(h = c(0:4), col = 'grey')
    abline(h = log10(c(1:9)*1), col = 'grey', lty = 2)
    abline(h = log10(c(1:9)*10), col = 'grey', lty = 2)
    abline(h = log10(c(1:9)*100), col = 'grey', lty = 2)
    abline(h = log10(c(1:9)*1000), col = 'grey', lty = 2)
    abline(h = log10(c(1:9)*10000), col = 'grey', lty = 2)
    axis(2, at = c(0:4), labels = 10^c(0:4))
    
    points(log10(counts$count), pch = 15)
    mtext(text = currentName,
          side = 3,
          line = -1.5,
          adj = 0.01,
          cex = 1.5)
    mtext(text = paste(nrow(predictions), ' ROIs'),
          adj = 0.99,
          line = -1.5,
          cex = 1.5)
    
    dev.off()
  }
}
