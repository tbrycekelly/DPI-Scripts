source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

outputDir = '../../export/pull'
nMax = 1000 # Maximum number of images per pull folder
pMin = 0.0 # minimum probability for pulled images (0 = pull all, 0.9 = 90% confidence)

camera = 'camera0'
transect = 'test1'
segmentationName = 'REG'
modelName = 'iota121v1'


#### Autopilot from here:

sourceFiles = list(
  rawPath = paste0('../../raw/camera0/', transect, '/'),
  segmentationPath = paste0('../../analysis/', camera, '/segmentation/', transect, '-', segmentationName, '/'),
  classificationPath = paste0('../../analysis/', camera, '/classification/', transect, '-', segmentationName, '-', modelName, '/'),
  modelPath = '../model'
)

sourceFiles$segmentationFiles = list.files(sourceFiles$segmentationPath, pattern = 'statistics.csv')
sourceFiles$classificationFiles = list.files(sourceFiles$classificationPath, pattern = 'prediction.csv')
sourceFiles$rawFiles = list.files(sourceFiles$rawPath, pattern = '.avi')


if (!dir.exists(outputDir)) {
  dir.create(outputDir, recursive = T)
}

for (i in 1:length(sourceFiles$classificationFiles)) {
  predictions = read.csv(paste0(sourceFiles$classificationPath, sourceFiles$classificationFiles[i]), header = T)
  
  ## Extract ROIs
  zipName = paste0(sourceFiles$segmentationPath, gsub('_prediction.csv', '.zip', sourceFiles$classificationFiles[i]))
  extractName = paste0(sourceFiles$segmentationPath, gsub('_prediction.csv', '/', sourceFiles$classificationFiles[i]))
  unzip(zipfile = zipName, exdir = extractName)
  
  classes = colnames(predictions)[-1]
  pmax = apply(predictions[,-1], 1, max)
  pmax = round(pmax, digits = 2)
  pIndex = apply(predictions[,-1], 1, which.max)
  
  for (class in classes) {
    if (!dir.exists(paste0(outputDir, '/', class))) {
      dir.create(paste0(outputDir, '/', class), recursive = T)
    }
  }
  
  l = pmax >= pMin
  
  file.copy(from = paste0(extractName, '/', predictions$X)[l], 
            to = paste0(outputDir, '/', classes[pIndex], '/[', pmax, '] ', gsub('.png', '', predictions$X), '.png')[l])
}
unlink(extractName, recursive = T)

## Subdivide large folders
nMax = 200

for (folder in list.dirs(outputDir)[-1]) {
  pulledRois = sample(list.files(folder, pattern = '.png'))
  n = length(pulledRois)
  nFolders = ceiling(n / nMax)
  
  if (nFolders > 1) {
    message('Making subfolder for ', folder)
    for (nCurrent in 1:nFolders) {
      dir.create(paste0(folder, '/', nCurrent, '/'))
      index = ((nCurrent - 1) * nMax + 1):min(nCurrent * nMax, length(pulledRois))
      file.copy(from = paste0(folder, '/', pulledRois[index]),
                to = paste0(folder, '/', nCurrent, '/', pulledRois[index]))
      file.remove(paste0(folder, '/', pulledRois[index])) # Clean up
    }
  }
  
}



