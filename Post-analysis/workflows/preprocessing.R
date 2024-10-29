library(PlanklinePS)
library(ini)

args = commandArgs(trailingOnly = TRUE)
config = ini::read.ini(args[1])
path = args[2]

if (!'dt' %in% names(config$R) | !'p_threshold' %in% names(config$R)) {
  stop('Config file is missing a required parameter for preprocessing.')
}

## parameters
frame.rate = 20.5
bin.width = as.numeric(config$R$dt) # second(s)
p.threshold = as.numeric(config$R$p_threshold) # probability threshold

## File names
tmp = strsplit(x = path, split = '/')[[1]]
file = tmp[length(tmp)]
path = gsub(file, '', path)
base.dir = gsub('/classification/', '', path)

measure.file = paste0(base.dir, '/segmentation/', gsub('.csv', '.tar', file))


out.dir = paste0(base.dir, '/R/')
if (!dir.exists(out.dir)) {
  dir.create(out.dir)
  dir.create(paste0(out.dir, '/classification/'))
  dir.create(paste0(out.dir, '/measurement/'))
  dir.create(paste0(out.dir, '/bin/'))
}

class = load.classifications(path, F)
saveRDS(class, file = paste0(out.dir, '/classification/', class$image[1], ' classification.rds'))

summary = load.measurements(measure.file, F)
saveRDS(summary, file = paste0(out.dir, '/measurement/', summary$image[1], ' measurement.rds'))


## Add timestamp
temp = strsplit(class$image[1], split = '-')[[1]]
class$time = as.POSIXct(paste0(temp[3], '-', temp[4], '-', temp[5], ' ', temp[6], ':', temp[7], ':', temp[8]), tz = 'UTC') + (class$frame - 1) / frame.rate

## bin
class$time = as.POSIXct(round(as.numeric(class$time) / bin.width) * bin.width, origin = '1970-01-01')
bin = data.frame(time = unique(class$time))

bin$frames = sapply(bin$time, function(x) {diff(range(class$frame[class$time == x]))})

for (n in unique(class$class)) { ## Already normalized to number of frames!
  bin[[n]] = sapply(bin$time, function(x) {sum(class$time == x & class$class == n & class$p >= p.threshold)}) / bin$frames 
}

## output
saveRDS(bin, file = paste0(out.dir, '/bin/', class$image[1], ' bin.rds'))