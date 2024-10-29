#' @export
load.acs.files = function(files, dt = 1, min.size = 1024, verboes = T) {
  
  #### Filter files by size
  file = data.frame(files = files, keep = T)
  
  ## Remove files smaller than 1kB
  for (i in 1:nrow(file)) {
    if (file.size(file$file[i]) < min.size) {
      file$keep[i] = F
    }
  }
  
  message(' Identified ', sum(file$keep), ' file(s) of valid size and ', sum(!file$keep), ' file(s) of invalid size.')
  files = file$files[file$keep]
  
  ## Load in first file.
  message(' Attempting to load file 1... ', appendLF = F)
  stamp = strsplit(x = files[1], split = '/')[[1]]
  stamp = stamp[length(stamp)]
  stamp = strsplit(stamp, '_')[[1]]
  stamp = gsub('.dat', '', stamp[length(stamp)])
  start.time = TheSource::make.time(year = substr(stamp, 1, 4),
                         month = substr(stamp, 5, 6),
                         day = substr(stamp, 7, 8),
                         hour = substr(stamp, 9, 10),
                         minute = substr(stamp, 11, 12),
                         second = substr(stamp, 13, 14),
                         tz = 'UTC')
  dat = as.data.frame(data.table::fread(files[1], skip = 99, verbose = F, showProgress = F))
  dat$Time = (dat$`Time(ms)` - min(dat$`Time(ms)`))/1000 + start.time
  dat$`Time(ms)` = NULL
  message('Done.')
  
  ## Read in rest of data
  for (i in 2:length(files)) {
    message(' Attempting to load file ', i, '... ', appendLF = F)
    temp = as.data.frame(data.table::fread(files[i], skip = 99, verbose = F, showProgress = F))
    
    ## Time
    stamp = strsplit(x = files[i], split = '/')[[1]]
    stamp = stamp[length(stamp)]
    stamp = strsplit(stamp, '_')[[1]]
    stamp = gsub('.dat', '', stamp[length(stamp)])
    start.time = TheSource::make.time(year = substr(stamp, 1, 4),
                           month = substr(stamp, 5, 6),
                           day = substr(stamp, 7, 8),
                           hour = substr(stamp, 9, 10),
                           minute = substr(stamp, 11, 12),
                           second = substr(stamp, 13, 14),
                           tz = 'UTC')
    temp$Time = (temp$`Time(ms)` - min(temp$`Time(ms)`))/1000 + start.time
    temp$`Time(ms)` = NULL
    
    if (ncol(temp) != ncol(dat)) {
      message('Invalid number of columns (', ncol(temp), ' vs ', ncol(dat), '), skipping.')
    } else {
      dat = rbind.data.frame(dat, temp)
      message('Done.')
    }
  }
  
  message('Finished.')
  dat
}
