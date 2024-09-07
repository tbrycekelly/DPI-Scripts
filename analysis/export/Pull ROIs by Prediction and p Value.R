source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

outputDir = '../../export/testWithPrior'
nMax = 1000 # Maximum number of images per pull folder
pMin = 0.0 # minimum probability for pulled images (0 = pull all, 0.9 = 90% confidence)

camera = 'camera1'
transect = 'SewardLine_Summer22'
segmentationName = 'REG'
modelName = 'lambda121v2'


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
  
  classes = colnames(predictions)[-1]
  pmax = apply(predictions[,-1] * weight, 1, function(x){max(x) / sum(x)})
  pmax = round(pmax, digits = 2)
  pIndex = apply(predictions[,-1] * weight, 1, which.max) # Weighted based on prior
  
  for (class in classes) {
    if (!dir.exists(paste0(outputDir, '/', class))) {
      dir.create(paste0(outputDir, '/', class), recursive = T)
    }
  }
  
  l = pmax >= pMin
  
  file.copy(from = paste0(extractName, '/', predictions$X)[l], 
            to = paste0(outputDir, '/', classes[pIndex], '/[', roundNumber(pmax), '] ', gsub('.png', '', predictions$X), '.png')[l])
  unlink(extractName, recursive = T)
  endTime = Sys.time()
  
  message('Finished file ', i, ' out of ', length(sourceFiles$classificationFiles),
          '.\tElapsed time: ', round(as.numeric(endTime) - as.numeric(initTime)), ' sec',
          '\tRemaining time: ', round((as.numeric(endTime) - as.numeric(initTime)) / i * (length(sourceFiles$classificationFiles) - i) / 60), ' min')
}


## Subdivide large folders
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
