#### segmentation

#' @title Load Measurements file
#' @param path filepath to measurement file to load.
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


roundNumber = function(x, digits = 2) {
  out = rep(NA, length(x))
  x = round(x, digits = digits)
  
  for (i in 1:length(x)) {
    dn = max(digits - nchar(x[i])+2, 0)
    out[i] = paste0(x[i], paste0(rep(0, dn), collapse = ''))
  }
  
  ## Return
  out
}


padNumber = function(x, digits = 2) {
  out = rep(NA, length(x))
  x = round(x, digits = digits)
  
  for (i in 1:length(x)) {
    dn = max(digits - nchar(x[i]), 0)
    out[i] = paste0(paste0(rep(0, dn), x[i], collapse = ''))
  }
  
  ## Return
  out
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


classify = function(prediction, weights = NULL) {
  
  if (is.null(weights)) {
    weights = 1
  }
  
  class = apply(prediction[,-c(1:4)], 1, function(x) {which.max(x * weights)})
  class = colnames(prediction)[-c(1:4)][class]
  
  cbind(prediction[,c(1:4)], data.frame(class = class))
}


getPrior = function(prediction) {
  stats = data.frame(class = colnames(prediction), n = NA, weight = NA)
  #stats$weight = apply(prediction, 2, sum)
  #stats$weight = stats$weight / sum(stats$weight)
  
  prediction = apply(prediction, 1, which.max)
  for (i in 1:nrow(stats)) {
    k = which(prediction == i)
    stats$n[i] = length(k)
  }
  stats$weight = stats$n / length(prediction) + 0.5 / nrow(stats)
  stats$weight = stats$weight / sum(stats$weight)
  stats
}


logModel = function(path) {
  
}

loadLog = function(path) {
  
}


pathConcat = function(...) {
  path = paste0(list(...), collapse = '')
  path = paste0(path, '/')
  path = gsub('//', '/', path)
  path
}

