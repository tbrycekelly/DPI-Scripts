feeds = c('../../sensor/SKQ202210S-Transect8/AD1216 Publisher/',
          '../../sensor/SKQ202210S-Transect8/Altimeter Publisher/',
          '../../sensor/SKQ202210S-Transect8/CTD 1 Publisher/',
          '../../sensor/SKQ202210S-Transect8/CTD 2 Publisher/',
          '../../sensor/SKQ202210S-Transect8/Flowmeter Publisher/',
          '../../sensor/SKQ202210S-Transect8/Fluorometer 1 Publisher/',
          '../../sensor/SKQ202210S-Transect8/Fluorometer 2 Publisher/',
          '../../sensor/SKQ202210S-Transect8/GPS Publisher/',
          '../../sensor/SKQ202210S-Transect8/Inclinometer Publisher/')

feedID = sapply(runif(6), function(x) digest::digest(x, algo = 'crc32'))

loadedData = list()

for (i in 1:length(feeds)) {
  loadedData[[i]] = list()
  
  for (f in list.files(path = feeds[i], pattern = '.csv', full.names = T)) {
    tmp = read.csv(f)
    loadedData[[i]][[f]] = tmp
  }
}


install.packages("RPostgres")  # if not already installed
library(DBI)
library(RPostgres)
library(jsonlite)

con <- dbConnect(
  RPostgres::Postgres(),
  dbname = "test",
  host = "localhost",     # or your IP address
  port = 5432,
  user = "postgres",
  password = "1024"
)


{
  a = Sys.time()
  
  for (i in 1:length(loadedData)) {
    message(i)
    for (j in 1:length(loadedData[[i]])) {
      #jsontmp = apply(loadedData[[i]][[j]], 1, toJSON)
      for (k in 1:nrow(loadedData[[i]][[j]])) {
        dbExecute(con, "
        INSERT INTO testtable5 (datetime, instrumentidentifier, data)
        VALUES ($1, $2, $3)
                  ",
                  params = list(as.POSIXct(loadedData[[i]][[j]][k,1]),
                                feedID[i],
                                runif(1)))
      }
    }
  }
  
  b = Sys.time()
}

dbDisconnect(con)
b-a # ~40 minutes on macbook pro


library(mongolite)
m <- mongo(collection = "test3",
           db = "local",
           url = "mongodb://localhost")

{
  a = Sys.time()
  
  for (i in 1:length(loadedData)) {
    message(i)
    for (j in 1:length(loadedData[[i]])) {
      for (k in 1:nrow(loadedData[[i]][[j]])) {
        tmp = list(
          datetime = unbox(as.POSIXct(loadedData[[i]][[j]][k,1])),
          instrument = unbox(feedID[i]),
          data = unbox(loadedData[[i]][[j]][k,])
        )
        
        m$insert(toJSON(tmp, auto_unbox = T, POSIXt = 'mongo', pretty = F, raw = 'mongo'))
      }
    }
  }
  
  b = Sys.time()
}

jsonlite::toJSON(tmp, raw = 'mongo', pretty = T, POSIXt = 'mongo')
jsonlite::toJSON(tmp, auto_unbox = T, POSIXt = 'mongo', pretty = T)
jsonify::to_json(tmp)


m <- mongo(collection = "test2",
           db = "local",
           url = "mongodb://localhost")



{
  a = Sys.time()
  
  for (i in 1:length(loadedData)) {
    message(i)
    for (j in 1:length(loadedData[[i]])) {
      for (k in 1:nrow(loadedData[[i]][[j]])) {
        tmp = list(
          datetime = gsub(' ', 'T', paste0(as.POSIXct(loadedData[[i]][[j]][k,1], tz = 'UTC'), 'Z')),
          meta = list(
            cruise = 'SQK202410S',
            transect = 'Sewardline',
            startingLocation = c(58.4, -145.23),
            endingLocation = c(60.4, -146.1)
            ),
          instrument = feedID[i],
          data = loadedData[[i]][[j]][k,]
        )
        m$insert(toJSON(tmp, pretty = F, POSIXt = 'mongo', raw = 'mongo', auto_unbox = T))
      }
    }
  }
  
  b = Sys.time()
}
b-a


m <- mongo(collection = "testimg1",
           db = "local",
           url = "mongodb://localhost")

{
  a = Sys.time()
  
  for (i in 1:10) {
    message(i)
    #for (j in 1:1e5) {
      w = sample(1:200, 1, prob = 1 / (1:200))
      h = sample(1:200, 1, prob = 1 / (1:200))
      
      tmp = list(
        datetime = Sys.time(),
        meta = list(
          cruise = 'SQK202410S',
          transect = 'Sewardline',
          startingLocation = c(58.4, -145.23),
          endingLocation = c(60.4, -146.1),
          file = 'test.avi',
          frame = sample(1:1024, 1),
          roi = sample(1:1e4, 1),
          camera = 'camera0'
        ),
        width = w,
        height = h,
        imgdata = array(runif(w*h), dim = c(w,h))
      )
      for (j in 1:1e5) {
      ## Insert data record
      m$insert(toJSON(tmp, pretty = F, POSIXt = 'mongo', raw = 'mongo', auto_unbox = T))
    }
  }
  
  b = Sys.time()
}
b-a

