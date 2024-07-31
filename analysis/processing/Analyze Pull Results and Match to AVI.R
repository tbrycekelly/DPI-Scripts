source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

outputDir = '../../export/pull/'

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

roiIndex = data.frame(path = roiFiles,
                      class = NA,
                      filename = NA,
                      frame = NA,
                      roi = NA,
                      camera = NA,
                      transect = NA,
                      segmentationName = NA,
                      modelName = NA,
                      aviFile = NA,
                      time = NA,
                      width = NA,
                      height = NA)

roiFiles = gsub(outputDir, '', roiFiles)
roiFiles = strsplit(roiFiles, split = '/')
message('Found ', length(roiFiles), ' image files.')



for (i in 1:nrow(roiIndex)) {
  if (length(roiFiles[[i]]) > 1) {
    roiIndex$class[i] = roiFiles[[i]][1]
    tmp = strsplit(roiFiles[[i]][2], split = ' ')[[1]]
    roiIndex$filename[i] = gsub('.png', '', tmp[2])
    ## get frame and roi number
    tmp = strsplit(roiIndex$filename[i], split = '-')[[1]]
    roiIndex$frame[i] = as.numeric(tmp[1])
    roiIndex$roi[i] = as.numeric(tmp[2])
    tmp = png::readPNG(source = roiIndex$path[i])
    roiIndex$height[i] = dim(tmp)[1]
    roiIndex$width[i] = dim(tmp)[2]
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
statistics = statistics[statistics$w + statistics$h > 50, ] # only want to actually look at the larger ROIs (perimeter > 100).
statistics$size = apply(statistics[,c('w','h')], 1, max)

for (i in 1:nrow(roiIndex)) {
  if (i %% 1000 == 0) {
    message('.')
  }
  k = which(roiIndex$frame[i] == statistics$frame & roiIndex$roi[i] == statistics$crop & roiIndex$width[i] == statistics$size)
  
  if (length(k) > 1) {
    message('Name collision found (n=', length(k), ').')
  } else if (length(k) == 1) {
    roiIndex$aviFile[i] = statistics$filename[k]
    #statistics = statistics[-k,]
  } else {
    message('No matching particle information found for ', roiIndex$filename[i])
  }
}
