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
message('Found ', length(roiFiles), ' image files.')

roiIndex = data.frame(path = roiFiles,
                      class = NA,
                      filename = NA,
                      frame = NA,
                      roi = NA,
                      camera = camera,
                      transect = transect,
                      segmentationName = NA,
                      modelName = modelName,
                      aviFile = NA,
                      time = NA,
                      width = NA,
                      height = NA)

roiFiles = gsub(outputDir, '', roiFiles)
roiFiles = strsplit(roiFiles, split = '/')


for (i in 1:nrow(roiIndex)) {
  if (length(roiFiles[[i]]) > 1) {
    n = length(roiFiles[[i]])
    roiIndex$class[i] = roiFiles[[i]][n-1]
    
    tmp = strsplit(roiFiles[[i]][n], split = ' ')[[1]]
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

View(roiIndex)

## Load segmentation statistics files:
statistics = loadMeasurement(paste0(sourceFiles$segmentationPath, '/', sourceFiles$segmentationFiles[1]))
statistics$size = apply(statistics[,c('w','h')], 1, max)
statistics = statistics[statistics$size >= min(roiIndex$width, na.rm = T), ] # only want to actually look at the larger ROIs (perimeter > 100).

if (length(sourceFiles$segmentationFiles) > 1) {
  for (i in 2:length(sourceFiles$segmentationFiles)) {
    message('Loading statistics file ', i)
    tmp = loadMeasurement(paste0(sourceFiles$segmentationPath, '/', sourceFiles$segmentationFiles[i]))
    tmp$size = apply(tmp[,c('w','h')], 1, max)
    tmp = tmp[tmp$size >= min(roiIndex$width, na.rm = T), ] # only want to actually look at the larger ROIs (perimeter > 100).
    
    statistics = rbind(statistics, tmp)
  }
}

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

saveRDS(roiIndex, file = paste0(outputDir, '/roiIndex.rds'))
