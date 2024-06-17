#' @title Load DPI Publisher Files
#' @export
load.dpi.files = function(files, skip = 1, min.size = 1024, verbose = T) {
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
  message('Done.')
  
  ## Read in rest of data
  for (i in 2:length(files)) {
    message(' Attempting to load file ', i, '... ', appendLF = F)
    temp = data.table::fread(files[i], skip = skip, fill = T, verbose = F, showProgress = F)
    
    if (ncol(temp) != ncol(dat)) {
      message('Invalid number of columns (', ncol(temp), ' vs ', ncol(dat), '), skipping.')
    } else {
      dat = rbind.data.frame(dat, temp, fill = T)
      message('Done.')
    }
  }
  
  message(' Setting up datetimes and ordering files.')
  colnames(dat)[1] = 'Time'
  
  ## Make and sort by time
  dat$Time = as.POSIXct(dat$Time, origin = TheSource::make.time(1904))
  dat = dat[order(dat$Time),]
  
  message('Finished.')
  as.data.frame(dat)
}