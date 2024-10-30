library(RSQLite)
source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

outputDir = '../../export/'

camera = 'camera0'
transect = 'test1'
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

message('Found ', length(sourceFiles$segmentationFiles), ' segmentation results.')
message('Found ', length(sourceFiles$classificationFiles), ' classification results.')

preprocessSegmentationToSQLite(sourceFiles$segmentationPath, sourceFiles$segmentationFiles, paste0(outputDir, '/database.db'), 'segmentationRaw')
preprocessClassificationToSQLite(sourceFiles$classificationPath, sourceFiles$classificationFiles, paste0(outputDir, '/database.db'), 'classificationRaw')



preprocessSegmentationToSQLite = function(path, files, db_path, table_name = 'segmentationRaw') {
  
  conn = dbConnect(RSQLite::SQLite(), dbname = db_path)
  
  for (file in files) {
    message('Preprocessing file ', file, '.')
    # Read CSV file
    data = read.csv(paste0(path, '/', file))
    data$file = gsub('.avi statistics.csv', '', file)
    dbWriteTable(conn, name = table_name, value = data, row.names = FALSE, append = T)
  }
  dbDisconnect(conn)
}



preprocessClassificationToSQLite = function(path, files, db_path, table_name = classificationRaw) {
  
  conn = dbConnect(RSQLite::SQLite(), dbname = db_path)
  
  for (file in files) {
    message('Preprocessing file ', file, '.')
    # Read CSV file
    data = read.csv(paste0(path, '/', file))
    data$file = gsub('.avi_prediction.csv', '', file)
    tmp = gsub('.png', '', data$X)
    tmp = strsplit(tmp, '-')
    data$frame = as.numeric(sapply(tmp, function(x){x[7]}))
    data$roi = as.numeric(sapply(tmp, function(x){x[8]}))
    colnames(data)[1] = 'image'
    
    dbWriteTable(conn, name = table_name, value = data, row.names = FALSE, append = T)
  }
  dbDisconnect(conn)
}

