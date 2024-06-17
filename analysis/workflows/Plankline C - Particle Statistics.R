library(PlanklinePS)
library(TheSource)

base.dir = '/media/plankline/Data/Data/SNR10/'
transects = readRDS('/media/plankline/Data/Sensor/SKQJ2022_data/_rdata/transects.rds')
transect.times = sapply(c(1:length(transects)), function(x) {mean(transects[[x]]$Time, na.rm = T)})

## Determine what sesnor data goes with these files:
transect.n = which.min((as.numeric(mean(bin$time, na.rm = T)) - transect.times)^2)
sensor = transects[[transect.n]]
sensor$Time = sensor$Time - 3600 * 8 ## Adj timezone

## Measurement files
measure.files = list.files(paste0(base.dir, '/R/measurement/'), pattern = '.rds', full.names = T)

summary = data.frame(file = measure.files, n = NA, area = NA)

for (i in 1:nrow(summary)) {
  measure = readRDS(measure.files[i])
  summary$n[i] = nrow(measure)
  summary$area[i] = mean(measure$area)
}

write.csv(summary, file = paste0(base.dir, '/R/measurement/statistics.csv'), row.names = F)
