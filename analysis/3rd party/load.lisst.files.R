#' @title Load LISST Files
#' @export
load.lisst.files = function(files, min.size = 1024, verbose = T) {
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
  dat = data.table::fread(files[1], fill = T, verbose = F, showProgress = F)
  message('Done.')
  
  ## Read in rest of data
  for (i in 2:length(files)) {
    message(' Attempting to load file ', i, '... ', appendLF = F)
    temp = data.table::fread(files[i], fill = T, verbose = F, showProgress = F)
    
    if (ncol(temp) != ncol(dat)) {
      message('Invalid number of columns (', ncol(temp), ' vs ', ncol(dat), '), skipping.')
    } else {
      dat = rbind.data.frame(dat, temp, fill = T)
      message('Done.')
    }
  }
  
  message(' Setting up datetimes and ordering files.')
  
  colnames(dat) = c(paste0('VC', 1:36),
                    'Laser.mW',
                    'Voltage',
                    'Analog.1',
                    'Laser.Ref',
                    'Depth',
                    'Temperature',
                    'Year', 'Month', 'Day', 'Hour', 'Minute', 'Seconds',
                    'Analog.2',
                    'Mean.Diameter', 'PPM',
                    'Humidity',
                    'X', 'Y', 'Z',
                    'Raw.Pressure.1', 'Raw.Pressure.2',
                    'Light',
                    'Analog.3',
                    'Transmission', 'Attenuation')
  
  ## Make and sort by time
  dat$Time = TheSource::make.time(dat$Year, dat$Month, dat$Day, dat$Hour, dat$Minute, dat$Seconds, tz = 'GMT')
  dat = dat[order(dat$Time),]
  
  message('Finished.')
  as.data.frame(dat)
}
