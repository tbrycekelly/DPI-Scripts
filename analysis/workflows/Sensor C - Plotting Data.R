library(PlanklinePS)
library(TheSource)

## Setup
save.dir = '/media/plankline/Data/Sensor/SKQJ2022_data/_rdata/'
transects = readRDS('/media/plankline/Data/Sensor/SKQJ2022_data/_rdata/transects.rdata')

#### MAPS

## CRP
pdf(paste0(save.dir, 'SKQ202210S Maps.pdf'))
map = make.map2(coast = 'coastline4',
                lon = -147,
                lat = 59.5,
                scale = 200,
                p = make.proj('nsper', lat= 50, lon = -145),
                dlon = 2,
                dlat = 2,
                land.col = 'black')

for (i in 1:length(transects)) {
  col = pals::alphabet(length(transects))[i]
  add.map.line(map, lon = transects[[i]]$Longitude, transects[[i]]$Latitude, greatCircle = F, col = col, lwd = 5)
  add.map.text(map,
               text = i,
               lon = mean(transects[[i]]$Longitude, na.rm = T),
               lat = mean(transects[[i]]$Latitude, na.rm = T),
               pos = 2,
               col = col)
}


for (i in 1:length(transects)) {
  map = make.map2(coast = 'coastline3',
                  lon = -147,
                  lat = 59.5,
                  scale = 200,
                  p = make.proj('nsper', lat= 50, lon = -145),
                  dlon = 2,
                  dlat = 2,
                  land.col = 'black')
  
  col = pals::alphabet(length(transects))[i]
  add.map.line(map, lon = transects[[i]]$Longitude, transects[[i]]$Latitude, greatCircle = F, col = col, lwd = 5)
  add.map.text(map,
               text = i,
               lon = mean(transects[[i]]$Longitude, na.rm = T),
               lat = mean(transects[[i]]$Latitude, na.rm = T),
               pos = 2,
               col = col)
}
dev.off()

browseURL(paste0(save.dir, 'SKQ202210S Maps.pdf'))


for (i in 1:length(transects)) {
  pdf(paste0(save.dir, 'Section Plots Preliminary (transect ', 1,').pdf'))
  par(mfrow = c(2,2))
  
  for (j in 1:ncol(transects[[i]])) {
    plot(transects[[i]]$Latitude, transects[[i]]$Depth,
         pch = '.',
         col = make.pal(transects[[i]][[j]], pal = 'inferno'),
         ylim = c(100, 0),
         main = names(transects[[i]])[j])
  }
  
  dev.off()
  browseURL(paste0(save.dir, 'Section Plots Preliminary (transect ', 1,').pdf'))
}

#### Build SECTIONS

section = list()

for (i in 1:length(transects)) {
  message(i)
  fields = names(transects[[i]])
  fields = fields[!fields %in% c('Distance', 'Depth')]
  section[[i]] = build.section(x = transects[[i]]$Distance,
                               y = transects[[i]]$Depth,  
                               z = transects[[i]][,fields],
                               field.names = fields,
                               ylim = c(0,200),
                               nx = 150,
                               ny = 200,
                               y.factor = 10,
                               uncertainty = 1,
                               neighborhood = 20,
                               lat = transects[[i]]$Latitude,
                               lon = transects[[i]]$Longitude,
                               gridder = gridODV,
                               verbose = F)
}


for (i in 1:length(transects)) {
  pdf(paste0(save.dir, 'Section Plots (transect ', i,').pdf'))
  par(mfrow = c(2,1))
  for (n in 1:ncol(section[[i]]$grid)) {
    plot.section(section[[i]], field = n,
                 pal = 'inferno',
                 ylim = c(200,0),
                 mark.points = F,
                 include.cex = 0.2,
                 ylab = 'Depth',
                 xlab = 'Section Distance',
                 main = paste0('Transect ', i, ' - ', names(section[[i]]$grid)[n]))
  }
  
  dev.off()
}

browseURL(paste0(save.dir, 'Section Plots (transect ', 1,').pdf'))
