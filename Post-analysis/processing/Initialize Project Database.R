library(RSQLite)
source('processing/low level utilities.R')
source('processing/mid level utilities.R')
source('processing/database utilities.R')


#### User input: 

outputDir = '../../database'

camera = 'camera0'
transect = 'test1'
segmentationName = 'REG'
modelName = 'kappa121v20'

#### Autopilot from here:

databaseName = paste0(camera, '-', transect, '-', segmentationName, '.db')

sourceFiles = list(
  rawPath = paste0('../../raw/camera0/', transect, '/'),
  segmentationPath = paste0('../../analysis/', camera, '/segmentation/', transect, '-', segmentationName, '/'),
  classificationPath = paste0('../../analysis/', camera, '/classification/', transect, '-', segmentationName, '-', modelName, '/'),
  modelPath = '../model'
)

sourceFiles$segmentationFiles = list.files(sourceFiles$segmentationPath, pattern = 'statistics.csv')
sourceFiles$classificationFiles = list.files(sourceFiles$classificationPath, pattern = 'prediction.csv')

message('Found ', length(sourceFiles$segmentationFiles), ' segmentation results.')
message('Found ', length(sourceFiles$classificationFiles), ' classification results.')

## Setup Output folder
if (!dir.exists(outputDir)) {
  dir.create(outputDir, recursive = T)
  message('Created output directory: "', outputDir, '".')
}

## Start with segmentation files (statistics csv's)
preprocessSegmentationToSQLite(paste0(outputDir, '/', databaseName),
                               'segmentation',
                               sourceFiles$segmentationPath,
                               sourceFiles$segmentationFiles)

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

for (i in 1:5) {
  weight = matrix(prior$weight, nrow = 1, ncol = length(prior$weight), byrow = T)
  prior = getPrior(classification[,-1] * weight)
}

preprocessClassificationToSQLite(db_path = paste0(outputDir, '/', databaseName),
                                 path = sourceFiles$classificationPath,
                                 files = sourceFiles$classificationFiles[-c(1:2)],
                                 prior$weight
                                 )



mergeQuery = "
CREATE TABLE matched AS
SELECT 
    segmentation.source,
    segmentation.frame,
    segmentation.roi,
    segmentation.datetime,
    segmentation.x,
    segmentation.y,
    segmentation.w,
    segmentation.h,
    segmentation.major_axis,
    segmentation.minor_axis,
    segmentation.area,
    segmentation.file,
    classification.image,
    classification.class1,
    classification.class2,
    classification.class3,
    classification.class1prob,
    classification.class2prob,
    classification.class3prob
FROM
    segmentation
INNER JOIN
    classification
ON
    segmentation.source = classification.source AND segmentation.frame = classification.frame AND segmentation.roi = classification.roi;
"

conn = dbConnect(RSQLite::SQLite(), dbname = paste0(outputDir, '/', databaseName))
result = dbExecute(conn, mergeQuery)
dbDisconnect(conn)

