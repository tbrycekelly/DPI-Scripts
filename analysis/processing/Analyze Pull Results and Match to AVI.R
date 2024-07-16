source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

outputDir = '../../export/hannah_qc/SewardLine_Summer22_camera1_iota201v1/'

camera = 'camera1'
transect = 'SewardLine_Summer22'
segmentationName = 'REG'
modelName = 'iota201v1'


#### Autopilot from here:

sourceFiles = list(
  rawPath = paste0('../../raw/', camera, '/', transect, '/'),
  segmentationPath = paste0('../../analysis/', camera, '/segmentation/', transect, '-', segmentationName, '/'),
  classificationPath = paste0('../../analysis/', camera, '/classification/', transect, '-', segmentationName, '-', modelName, '/'),
  modelPath = '../model'
)

sourceFiles$segmentationFiles = list.files(sourceFiles$segmentationPath, pattern = 'statistics.csv')
sourceFiles$classificationFiles = list.files(sourceFiles$classificationPath, pattern = 'prediction.csv')


## Load in list of all files in directory:
roiFiles = list.files(outputDir, pattern = '.png', full.names = T, recursive = T)
roiFiles = gsub(outputDir, '', roiFiles)
roiFiles = strsplit(roiFiles, split = '/')
message('Found ', length(roiFiles), ' image files.')

roiIndex = data.frame(class = rep(NA, length(roiFiles)),
                      filename = NA,
                      frame = NA,
                      roi = NA,
                      camera = NA,
                      transect = NA,
                      segmentationName = NA,
                      modelName = NA,
                      aviFile = NA,
                      time = NA)

for (i in 1:nrow(roiIndex)) {
  if (length(roiFiles[[i]]) == 2) {
    roiIndex$class[i] = roiFiles[[i]][1]
    tmp = strsplit(roiFiles[[i]][2], split = ' ')[[1]]
    roiIndex$filename[i] = gsub('.png', '', tmp[2])
    ## get frame and roi number
    tmp = strsplit(roiIndex$filename[i], split = '-')[[1]]
    roiIndex$frame = as.numeric(tmp[1])
    roiIndex$roi = as.numeric(tmp[2])
  }
}

## Load segmentation statistics files:
statistics = loadMeasurement(paste0(sourceFiles$segmentationPath, '/', sourceFiles$segmentationFiles[1]))
if (length(sourceFiles$segmentationFiles) > 1) {
  for (i in 2:length(sourceFiles$segmentationFiles)) {
    tmp = loadMeasurement(paste0(sourceFiles$segmentationPath, '/', sourceFiles$segmentationFiles[i]))
    statistics = rbind(statistics, tmp)
  }
}

for (i in 1:nrow(roiIndex)) {
  if (i %% 1000 == 0) {
    message('.')
  }
  k = which(roiIndex$frame[i] == statistics$frame & roiIndex$roi[i] == statistics$crop)
  
  if (length(k) > 1) {
    message('Name collision found (n=', length(k), ').')
  }
  if (length(k) == 1) {
    roiIndex$aviFile[i] = statistics$filename[k]
    #statistics = statistics[-k,]
  }
}
