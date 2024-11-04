
#### Segmentation

preprocessSegmentationToSQLite = function(db_path, table_name = 'segmentation', path, files) {
  
  conn = dbConnect(RSQLite::SQLite(), dbname = db_path)
  
  if (!'segmentation' %in% dbListTables(conn)) {
    
    createTableQuery = '
    CREATE TABLE segmentation (
    "source" TEXT,
    "frame" INTEGER,
    "roi" INTEGER,
    "datetime" REAL,
  	"x" INTEGER,
    "y" INTEGER,
    "w" INTEGER,
    "h" INTEGER,
    "major_axis" REAL,
    "minor_axis" REAL,
    "area" REAL,
    "file" TEXT,
    PRIMARY KEY (source, frame, roi)
  );'
    
    result <- dbExecute(conn, createTableQuery)
  }
  
  for (file in files) {
    message('Preprocessing file ', file, '.')
    # Read CSV file
    data = read.csv(paste0(path, '/', file))
    colnames(data)[colnames(data) == 'crop'] = 'roi'
    data$source = gsub(' statistics.csv', '', file)
    dbWriteTable(conn, name = table_name, value = data, row.names = FALSE, append = T)
  }
  dbDisconnect(conn)
}



loadTable = function(db_path, table_name = NULL) {
  
  conn = dbConnect(RSQLite::SQLite(), dbname = db_path)
  
  result = NULL
  
  if (is.null(table_name)) {
    message('No table specified. Availabile tables are: ')
    message(dbListTables(conn), collapse = ', ')
  } else {
    result = dbReadTable(conn, name = table_name)
  }
  dbDisconnect(conn)
  
  return(result)
}


#### PRIOR

savePrior = function(db_path, table_name = 'prior', model, prior) {
  
  prior$model = model
  
  conn = dbConnect(RSQLite::SQLite(), dbname = db_path)
  dbWriteTable(conn, name = table_name, value = prior, row.names = FALSE, append = T)
  dbDisconnect(conn)
}


#### Classifications

preprocessClassificationToSQLite = function(db_path, path, files, weight) {
  
  conn = dbConnect(RSQLite::SQLite(), dbname = db_path)
  
  if (!'classification' %in% dbListTables(conn)) {
    
    createTableQuery = '
    CREATE TABLE classification (
    "source" TEXT,
    "frame" INTEGER,
    "roi" INTEGER,
    "image" TEXT,
    "datetime" REAL,
  	"class1" INTEGER,
  	"class2" INTEGER,
  	"class3" INTEGER,
  	"class1prob" REAL,
  	"class2prob" REAL,
  	"class3prob" REAL,
    PRIMARY KEY (source, frame, roi)
  );'
    
    result <- dbExecute(conn, createTableQuery)
  }
  
  
  for (file in files) {
    message('Preprocessing file ', file, '.')
    # Read CSV file
    data = read.csv(paste0(path, '/', file))
    class = colnames(data)[-1]
    
    weight = matrix(weight, nrow = 1, ncol = length(weight), byrow = T)
    pTmp = t(apply(data[-1], 1, function(x) {(x * weight) / sum(x * weight)}))
    pOrder = t(apply(pTmp, 1, function(x){order(x, decreasing = T)}))[,1:3]
    
    data$file = gsub(' prediction.csv', '', file)
    tmp = gsub('.png', '', data$X)
    tmp = strsplit(tmp, '-')
    data$frame = as.numeric(sapply(tmp, function(x){x[9]})) ## TODO
    data$roi = as.numeric(sapply(tmp, function(x){x[10]})) ## TODO
    
    result = data.frame(source = data$file,
                        frame = data$frame,
                        roi = data$roi,
                        image = data[,1],
                        class1 = pOrder[,1],
                        class1prob = NA,
                        class2 = pOrder[,2],
                        class2prob = NA,
                        class3 = pOrder[,3],
                        class3prob = NA
    )
    
    for (i in 1:nrow(result)) {
      result$class1prob[i] = pTmp[i,result$class1[i]]
      result$class2prob[i] = pTmp[i,result$class2[i]]
      result$class3prob[i] = pTmp[i,result$class3[i]]
    }
    
    dbWriteTable(conn, name = 'classification', value = result, row.names = FALSE, append = T)
  }
  dbDisconnect(conn)
}

