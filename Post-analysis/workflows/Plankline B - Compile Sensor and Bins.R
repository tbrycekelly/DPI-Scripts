library(PlanklinePS)
library(TheSource)

base.dir = '/media/plankline/Data/Data/2022-07-22_13-39-05.528.Done/'
transects = readRDS('/media/plankline/Data/Sensor/SKQJ2022_data/_rdata/transects.rdata')
transect.times = sapply(c(1:length(transects)), function(x) {mean(transects[[x]]$Time, na.rm = T)})

#### Compile merged classification/sensor package dataset
bin.files = list.files(paste0(base.dir, '/R/bin/'), pattern = '.rds', full.names = T)
comp = data.frame(time = NA)

bin = readRDS(bin.files[round(length(bin.files)/2)]) ## Load one bin file to determine datetime
groups = sapply(strsplit(colnames(bin), split = '_'), function(x) {paste0(x[1], '_')})
groups = unique(groups)

## Determine what sesnor data goes with these files:
transect.n = which.min((as.numeric(mean(bin$time, na.rm = T)) - transect.times)^2)
sensor = transects[[transect.n]]
sensor$Time = sensor$Time - 3600 * 8 ## Adj timezone

for (i in 1:length(bin.files)) {
  message('Merging binned data and sensor data for file ', i, ' of ', length(bin.files), '...')
  bin = readRDS(bin.files[i])
  
  ## check if sensor data is available
  if (sum(sensor$Time >= min(bin$time) & sensor$Time <= max(bin$time)) < 1) {
    message('ERROR! No sensor data found for file ', bin.files[i],'.')
  }
  
  ## Merge sensor data with class file
  new.cols = colnames(sensor)[!colnames(sensor) %in% colnames(bin)]
  if (length(new.cols) > 0) {
    for (n in new.cols) {
      bin[[n]] = approx(sensor$Time, sensor[[n]], xout = bin$time, rule = 2)$y
    }
  }
  
  ## determine if any filler columns need to be added:
  new.cols = colnames(bin)[!colnames(bin) %in% colnames(comp)]
  if (length(new.cols) > 0) {
    for (n in new.cols) {
      comp[[n]] = NA
    }
  }
  
  new.cols = colnames(comp)[!colnames(comp) %in% colnames(bin)]
  if (length(new.cols) > 0) {
    for (n in new.cols) {
      bin[[n]] = NA
    }
  }
  
  comp = rbind(comp, bin)
}

## Cleanup and save
comp = comp[-1,]
comp = comp[order(comp$time),]
comp$Time = NULL
comp[comp == Inf] = 0
comp[is.na(comp)] = 0
comp = comp[, order(colnames(comp))]
comp$time = as.POSIXct(comp$time, origin = '1970-01-01')
saveRDS(comp, file = paste0(base.dir, '/R/Zooplankton Compilation.rds'))

{
  pdf(paste0(base.dir, '/R/Sample Plots.pdf'))
  par(mfrow = c(2,2))
  
  for (n in groups) {
    z = log10(group(comp, n))
    zlim = quantile(z, probs = c(0.02, 0.98), na.rm = T)
    plot(x = comp$time,
         y = comp$Depth,
         col = make.pal(z, min = 0, max = zlim[2], pal = 'parula'),
         pch = '.',
         ylim = rev(range(pretty(comp$Depth))),
         yaxs = 'i',
         main = paste('log', n))
  }
  
  
  
  plot(log10(comp$Chl[comp$Depth < 20]),
       log10(group(comp[comp$Depth < 20,], 'copepod_')),
       pch = 20,
       col = make.pal(comp$Depth[comp$Depth < 20],
                      pal = 'ocean.deep', rev = T),
       ylab = 'Copepods',
       xlab = 'Chl',
       xaxt = 'n',
       yaxt = 'n',
       main = 'Surface 20m')
  
  add.log.axis(1)
  add.log.axis(2)
  
  dev.off()
}

browseURL(paste0(base.dir, '/R/Sample Plots.pdf'))

## Start preliminary plot
colnames(comp) = paste0(colnames(comp), '_')
comp.reduced = data.frame(depth = comp$Depth_)

for (g in groups) {
  comp.reduced[[g]] = group(comp, pattern = g, verbose = F)
}

section = build.section(x = comp.reduced$time_,
                        y = comp.reduced$depth,
                        z = comp.reduced,
                        lat = comp.reduced$Latitude_,
                        lon = comp.reduced$Longitude_,
                        y.factor = 10,
                        neighborhood = 50,
                        uncertainty = 2,
                        nx = 200,
                        ny = 200,
                        ylim = c(5,100),
                        xlim = as.numeric(range(comp.reduced$time)),
                        field.names = colnames(comp.reduced))


pdf(paste0(base.dir, '/R/Preliminary Zooplankton Sections Transect ', transect.n,'.pdf'))
par(mfrow = c(2,1))
for (i in 3:ncol(section$grid)) {
  
  plot.section(section,
               field = i,
               ylim = c(100,0),
               pal = 'parula',
               zlim = range(pretty(quantile(section$grid[[i]], probs = c(0.02, 0.98), na.rm = T))),
               xaxt = 'n',
               ylab = 'Depth (m)',
               xlab = '')
  
  axis.POSIXct(1, comp.reduced$time)
  #add.section.contour(section, field = 'Temperature_', col = 'red')
  mtext(colnames(section$grid)[i], adj = 0)
}

dev.off()

browseURL(paste0(base.dir, '/R/Preliminary Zooplankton Sections Transect ', transect.n,'.pdf'))

