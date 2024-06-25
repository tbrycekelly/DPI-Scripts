

mergeMeasurementClassification = function(measurement,
                                      prediction) {
  
  measurement$class = NA
  
  for (i in 1:nrow(prediction)) {
    k = which(measurement$filename == prediction$sourcefile[i] & measurement$frame == prediction$frame[i] & measurement$crop == prediction$crop[i])
    measurement$class[k] = prediction$class[i]
  }
  
  measurement
}