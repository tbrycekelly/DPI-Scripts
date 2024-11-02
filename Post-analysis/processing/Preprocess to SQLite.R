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


RSQLite::dbReadTable(conn = conn, name = 'segmentationRaw')
RSQLite::dbGetQuery(conn = conn, 'select * from table segmentationRaw where frame < 10')


query <- "
SELECT
    segmentationRaw.id AS id1,
    segmentationRaw.frame,
    segmentationRaw.crop,
    segmentationRaw.X,
    classificationRaw.id AS id2
FROM
    segmentationRaw
INNER JOIN
    classificationRaw
ON
    segmentationRaw.frame = classificationRaw.frame AND segmentationRaw.crop = classificationRaw.roi;
"

# Execute the query and store the result in an R data frame
result <- dbGetQuery(conn, query)



conn = dbConnect(RSQLite::SQLite(), dbname = paste0(outputDir, '/database.db'))
createTableQuery = '
CREATE TABLE statistics (
  "source" TEXT,
  "frame" INTEGER,
  "roi" INTEGER,
  "x"	REAL,
	"y"	REAL,
	"w"	INTEGER,
	"h"	INTEGER,
	"major_axis"	REAL,
	"minor_axis"	REAL,
	"area"	REAL,
	"class1" INTEGER,
	"class2" INTEGER,
	"class3" INTEGER,
	"class1prob" REAL,
	"class2prob" REAL,
	"class3prob" REAL,
  PRIMARY KEY (source, frame, roi)
);
'
result <- dbExecute(conn, createTableQuery)
dbDisconnect(conn)

preprocessSegmentationToSQLite(sourceFiles$segmentationPath, sourceFiles$segmentationFiles, paste0(outputDir, '/database.db'), 'statistics')

data = data[1:10,]
data$class1 = 'test'
data$x = NULL
data$y = NULL
data$w = NULL
data$h = NULL

preprocessSegmentationToSQLite = function(path, files, db_path, table_name = 'segmentationRaw') {
  
  conn = dbConnect(RSQLite::SQLite(), dbname = db_path)
  
  for (file in files) {
    message('Preprocessing file ', file, '.')
    # Read CSV file
    data = read.csv(paste0(path, '/', file))
    colnames(data)[colnames(data) == 'crop'] = 'roi'
    data$source = gsub('.avi statistics.csv', '', file)
    dbWriteTable(conn, name = table_name, value = data, row.names = FALSE, append = T)
  }
  dbDisconnect(conn)
}


## primary key : 


updateEntries = function(db_path, table_name, newData) {
  conn = dbConnect(RSQLite::SQLite(), dbname = db_path)
  dbBegin(conn)
  for (i in 1:nrow(data)) {
    dbExecute(conn, 
              "UPDATE statistics SET class1 = ? WHERE source = ? AND frame = ? AND roi = ?", 
              params = list(data$class1[i], newData$source[i], newData$frame[i], newData$roi[i]))
  }
  dbCommit(conn)
  dbDisconnect(conn)
}