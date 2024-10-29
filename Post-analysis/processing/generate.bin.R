#' @export
generate.bin = function(dir, p.threshold = 0.5, frame.rate = 20.5, bin.width = 1) {
  
  ## First merge classification + measurements. Then load generate binned summary:
  class.file = list.files(dir,
                          pattern = '_prediction.rds',
                          full.names = T)
  
  for (i in 1:length(class.file)) {
    message('Binning file ', i, ' of ', length(class.file), '... ')
    class = readRDS(class.file[i])
    
    ## Add timestamp
    temp = strsplit(file, split = '/')[[1]]
    temp = temp[length(temp)]
    temp = strsplit(temp, split = '-|\\.')[[1]]
    class$time = as.POSIXct(paste0(temp[3], '-', temp[4], '-', temp[5], ' ', temp[6], ':', temp[7], ':', temp[8]), tz = 'UTC') + (class$frame - 1) / frame.rate
    
    ## bin
    class$time = as.POSIXct(round(as.numeric(class$time) / bin.width) * bin.width, origin = '1970-01-01')
    bin = data.frame(time = unique(class$time))
    
    bin$frames = sapply(bin$time, function(x) {diff(range(class$frame[class$time == x])) + 1})
    
    for (n in unique(class$class)) { ## Already normalized to number of frames!
      bin[[n]] = sapply(bin$time, function(x) {sum(class$time == x & class$class == n & class$p >= p.threshold)}) / bin$frames 
    }
    
    ## output
    saveRDS(bin, file = paste0(dir, gsub('_prediction', '_binnedprediction', class.file[i])))
  }
}