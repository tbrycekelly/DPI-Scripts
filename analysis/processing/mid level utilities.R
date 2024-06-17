

mergeMeasurementClassification = function(measurement,
                                      prediction) {
  
  measurement$class = NA
  
  for (i in 1:nrow(prediction)) {
    k = which(measurements$filename == prediction$sourcefile[i] & measurements$frame == prediction$frame[i] & measurements$crop == prediction$roi[i])
    measurement$class[k] = prediction$class[i]
  }
  
  measurement
}