#' @title Load DPI Publisher Files
#' @export
load.suna.files = function(files, skip = 21, min.size = 1024, verbose = T) {
  message('Starting to load in files, ', length(files), ' files specified.')
  
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
  dat = data.table::fread(files[1], skip = skip, fill = T)
  dat = data.frame(Time = make.time(year = substr(dat$V2, 1, 4)) + 86400 * as.numeric(substr(dat$V2, 5, 7)) + 3600 * dat$V3,
                   Nitrate = dat$V4)
  dat$Time = as.numeric(dat$Time)
  message('Done.')
  
  ## Read in rest of data
  for (i in 2:length(files)) {
    message(' Attempting to load file ', i, '... ', appendLF = F)
    temp = data.table::fread(files[i], skip = skip, fill = T, verbose = F, showProgress = F)
    temp = data.frame(Time = make.time(year = substr(temp$V2, 1, 4)) + 86400 * as.numeric(substr(temp$V2, 5, 7)) + 3600 * temp$V3,
                     Nitrate = temp$V4)
    temp$Time = as.numeric(temp$Time)
    
    if (ncol(temp) != ncol(dat)) {
      message('Invalid number of columns (', ncol(temp), ' vs ', ncol(dat), '), skipping.')
    } else {
      dat = rbind.data.frame(dat, temp, fill = T)
      message('Done.')
    }
  }
  
  ## Make and sort by time
  dat$Time = conv.time.unix(dat$Time)
  dat = dat[order(dat$Time),]
  
  message('Finished.')
  as.data.frame(dat)
}