#### segmentation

#' @export
loadMeasurement = function(path, verbose = T) {
  
  ## Get measurement file name
  f = strsplit(path, '/')[[1]]
  f = f[length(f)]
  f = gsub(' statistics.csv', '', f)
  f = gsub('.avi', '', f)
  f = gsub('.jpg', '', f)
  
  measurements = read.csv(path, header = T)
  for (i in 1:ncol(measurements)) {
    measurements[,i] = as.numeric(measurements[,i])
  }
  measurements = cbind(data.frame(filename = f), measurements)
  
  measurements
}



#### Predictions

splitRoiName = function(name) {
  name = gsub('.png', '', name)
  tmp = strsplit(name, split = '-')
  
  frame = as.numeric(sapply(tmp, function(x) {x[length(x)-1]}))
  crop = as.numeric(sapply(tmp, function(x) {x[length(x)]}))
  
  data.frame(frame = frame, crop = crop)
}


#' @title Load Classifications File
#' @import data.table
#' @export
loadPrediction = function(path, verbose = T) {
  
  filename = strsplit(path, split = '/')[[1]]
  filename = filename[length(filename)]
  filename = gsub('.avi_prediction.csv', '', filename)
  
  predictions = data.table::fread(path, header = T)
  predictions = as.data.frame(predictions)
  colnames(predictions)[1] = 'file'
  predictions = cbind(splitRoiName(predictions$file), predictions)
  predictions = predictions[order(predictions$crop),]
  predictions = predictions[order(predictions$frame),]
  predictions = cbind(data.frame(sourcefile = filename), predictions)
  
  predictions
}


classify = function(prediction) {
  
  class = apply(prediction[,-c(1:4)], 1, which.max)
  class = colnames(prediction)[-c(1:4)][class]
  
  cbind(prediction[,c(1:4)], data.frame(class = class))
}


logModel = function(path) {
  
}

loadLog = function(path) {
  
}