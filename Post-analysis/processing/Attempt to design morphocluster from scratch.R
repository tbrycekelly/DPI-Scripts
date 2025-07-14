source('processing/low level utilities.R')
source('processing/mid level utilities.R')

convert = function(x, pca) {
  #tmp = t(as.matrix(x))
  
  #result = pca$rotation %*% ((tmp - pca$center) / pca$scale)
  result = predict(pca, x)
  return(result)
}

determineCluster = function(x, centers) {
  cluster = rep(NA, ncol(x)) 
  
  for (i in 1:ncol(x)) {
    d = apply(centers, 1, function(center) {
      sum((x[,i] - center)^2)
    })
    
    cluster[i] = which.min(d)
  }
  
  return(cluster)
}

#### User input: 

outputDir = '../../export/features'
nMax = 1000 # Maximum number of images per pull folder

camera = 'camera0'
transect = 'test1-ffv1'
segmentationName = 'REG'
modelName = 'kappa121v20'


#### Autopilot from here:

sourceFiles = list(
  rawPath = paste0('../../raw/camera0/', transect, '/'),
  segmentationPath = paste0('../../analysis/', camera, '/segmentation/', transect, '-', segmentationName, '/'),
  classificationPath = paste0('../../analysis/', camera, '/classification/', transect, '-', segmentationName, '-', modelName, '/'),
  modelPath = '../model'
)

sourceFiles$segmentationFiles = list.files(sourceFiles$segmentationPath, pattern = 'statistics.csv')
sourceFiles$classificationFiles = list.files(sourceFiles$classificationPath, pattern = 'prediction.csv')

message('Found ', length(sourceFiles$classificationFiles), ' classification results.')

## Calculate priors from full dataset:
nSample = max(min(500, length(sourceFiles$classificationFiles)), length(sourceFiles$classificationFiles) * 0.2)
l = sample(1:length(sourceFiles$classificationFiles), size = nSample)
classification = read.csv(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[l[1]]), header = T)
for (i in 2:nSample) {
  message(round(i / nSample * 100), '%')
  tmp = read.csv(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[l[i]]), header = T)
  classification = rbind(classification, tmp)
}

## Calculate PCA rotation matrix then apply rotation matrix to data.
# Need to save rotation matrix to apply to future datasets as well.
problemCol = which(apply(classification[,-1], 2, sd) == 0)
for (col in problemCol) {
  message('Column ', col + 1, ' has zero variance, adding noise.')
  classification[,col + 1] = classification[,col + 1] + rnorm(nrow(classification), sd = 1e-8)
}

pca = prcomp(classification[,-1], tol = 1e-8, scale. = T, center = T)
classification[,-1] = convert(classification[,-1], pca)

## Determine cluster centers:
Ncenters = 100
centers = kmeans(classification[,-1], centers = Ncenters, iter.max = 100)$centers

if (!dir.exists(outputDir)) {
  dir.create(outputDir, recursive = T)
}

initTime = Sys.time()
for (i in 1:length(sourceFiles$classificationFiles)) {
  startTime = Sys.time()
  
  predictions = read.csv(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[i]), header = T)
  tmp = convert(predictions[,-1], pca)
  assignedCluster = determineCluster(tmp, centers = centers)
  
  
  ## Extract ROIs
  zipName = paste0(sourceFiles$segmentationPath, gsub(' prediction.csv', '.zip', sourceFiles$classificationFiles[i]))
  extractName = paste0(sourceFiles$segmentationPath, gsub(' prediction.csv', '/', sourceFiles$classificationFiles[i]))
  unzip(zipfile = zipName, exdir = extractName)
  
  classes = 1:Ncenters
  
  for (class in classes) {
    if (!dir.exists(paste0(outputDir, '/', class))) {
      dir.create(paste0(outputDir, '/', class), recursive = T)
    }
  }
  
  file.copy(from = paste0(extractName, '/', predictions$X), 
            to = paste0(outputDir, '/', assignedCluster, '/', gsub('.png', '', predictions$X), '.png'))
  unlink(extractName, recursive = T)
  endTime = Sys.time()
  
  message('Finished file ', i, ' out of ', length(sourceFiles$classificationFiles),
          '.\tElapsed time: ', round(as.numeric(endTime) - as.numeric(initTime)), ' sec',
          '\tRemaining time: ', round((as.numeric(endTime) - as.numeric(initTime)) / i * (length(sourceFiles$classificationFiles) - i) / 60), ' min')
}


;## Subdivide large folders
for (folder in list.dirs(outputDir)[-1]) {
  pulledRois = sample(list.files(folder, pattern = '.png'))
  n = length(pulledRois)
  nFolders = ceiling(n / nMax)
  
  if (nFolders > 1) {
    message('Making subfolder for ', folder, ' (x', nFolders, ')')
    for (nCurrent in 1:nFolders) {
      dir.create(paste0(folder, '/', nCurrent, '/'))
      index = ((nCurrent - 1) * nMax + 1):min(nCurrent * nMax, length(pulledRois))
      file.copy(from = paste0(folder, '/', pulledRois[index]),
                to = paste0(folder, '/', nCurrent, '/', pulledRois[index]))
      file.remove(paste0(folder, '/', pulledRois[index])) # Clean up
    }
  }
}

zip(outputDir, zipfile = paste0(outputDir, '.zip'))



## Example: fine a particular ROI png from the set of classification files.
for (i in 1:length(sourceFiles$classificationFiles)) {
  tmp = loadPrediction(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[i]))
  if (any('000339-002798.png' == tmp$file)) {
    print(i)
    break
  }
}

l = which(tmp$file == '000339-002798.png')
View(tmp[l,])
