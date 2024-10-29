library(openxlsx)
library(geosphere) # for distance calculations
library(PlanklinePS)

bin = function(x, y, xout, func = function (x) {mean(x, na.rm = T)}) {
  yout = rep(NA, length(xout))
  
  for (i in 1:length(xout)) {
    l = x == xout[i]
    if (sum(l) > 0) {
      yout[i] = func(y[l])
    }
  }
  
  yout
}

## Setup
save.dir = '/media/plankline/Data/Sensor/SKQJ2022_data/_rdata/'

gps = readRDS(paste0(save.dir, 'gps.rds'))
ctd1 = readRDS(paste0(save.dir, 'ctd1.rds'))
ctd2 = readRDS(paste0(save.dir, 'ctd2.rds'))
eng = readRDS(paste0(save.dir, 'engineering.rds'))
fluoro1 = readRDS(paste0(save.dir, 'fluorometer1.rds'))
fluoro2 = readRDS(paste0(save.dir, 'fluorometer2.rds'))
analog = readRDS(paste0(save.dir, 'analog.rds'))
acs = readRDS(paste0(save.dir, 'acs.rds'))
suna = readRDS(paste0(save.dir, 'suna.rds'))

## bin time

dt = 2 #seconds

gps$Time = conv.time.unix(round(as.numeric(gps$Time) / dt) * dt)
ctd1$Time = conv.time.unix(round(as.numeric(ctd1$Time) / dt) * dt)
ctd2$Time = conv.time.unix(round(as.numeric(ctd2$Time) / dt) * dt)
eng$Time = conv.time.unix(round(as.numeric(eng$Time) / dt) * dt)
fluoro1$Time = conv.time.unix(round(as.numeric(fluoro1$Time) / dt) * dt)
fluoro2$Time = conv.time.unix(round(as.numeric(fluoro2$Time) / dt) * dt)
analog$Time = conv.time.unix(round(as.numeric(analog$Time) / dt) * dt)
acs$Time = conv.time.unix(round(as.numeric(acs$Time) / dt) * dt)
suna$Time = conv.time.unix(round(as.numeric(suna$Time) / dt) * dt)

## Merging
dpi = data.frame(Time = unique(gps$Time))


## GPS
for (n in names(gps)[-1]) {
  dpi[[n]] = approx(as.numeric(gps$Time), gps[,n], xout = as.numeric(dpi$Time), ties = mean)$y
}

## CTD
for (n in names(ctd1)[-1]) {
  dpi[[n]] = approx(as.numeric(ctd1$Time), ctd1[[n]], xout = as.numeric(dpi$Time), ties = mean)$y
}
for (n in names(ctd2)[-1]) {
  dpi[[n]] = approx(as.numeric(ctd2$Time), ctd2[[n]], xout = as.numeric(dpi$Time), ties = mean)$y
}

## Analog
for (n in names(analog)[2:5]) {
  dpi[[n]] = approx(as.numeric(analog$Time), analog[[n]], xout = as.numeric(dpi$Time), ties = mean)$y
}

##Fluorometer
for (n in names(fluoro1)[c(3,5,7)]) {
  dpi[[n]] = approx(as.numeric(fluoro1$Time), fluoro1[[n]], xout = as.numeric(dpi$Time), ties = median)$y
}
for (n in names(fluoro2)[c(3,5,7)]) {
  dpi[[n]] = approx(as.numeric(fluoro2$Time), fluoro2[[n]], xout = as.numeric(dpi$Time), ties = median)$y
}

## ENG
for (n in names(eng)[2:6]) {
  dpi[[n]] = approx(as.numeric(eng$Time), eng[[n]], xout = as.numeric(dpi$Time), ties = mean)$y
}

## ACs
dpi[['a440']] = approx(as.numeric(acs$Time), acs$A440, xout = as.numeric(dpi$Time), ties = median)$y
dpi[['a675']] = approx(as.numeric(acs$Time), acs$A675.8, xout = as.numeric(dpi$Time), ties = median)$y

## SUNA
for (n in names(suna)[-1]) {
  dpi[[n]] = approx(as.numeric(suna$Time), suna[[n]], xout = as.numeric(dpi$Time), ties = median)$y
}


#### Filter
dpi = dpi[dpi$Pressure > 2,]

# Break at time gaps
k = c(0, which(diff(dpi$Time) > 3600), nrow(dpi))

transects = list()

for (i in 2:length(k)) {
  transects[[paste0('dpi', i-1)]] = dpi[(k[i-1]+1):k[i],]  
  
  ## Add section distance based on the cumulative sum of all the piecewise distances.
  transects[[paste0('dpi', i-1)]]$Distance = c(0,
                                               sapply(
                                                 c(2:nrow(transects[[paste0('dpi', i-1)]])),
                                                 function(x) {
                                                   geosphere::distCosine(p1 = c(transects[[paste0('dpi', i-1)]]$Longitude[x-1], transects[[paste0('dpi', i-1)]]$Latitude[x-1]),
                                                                         p2 = c(transects[[paste0('dpi', i-1)]]$Longitude[x], transects[[paste0('dpi', i-1)]]$Latitude[x]))
                                                 }) / 1e3)
  
  transects[[paste0('dpi', i-1)]]$Distance = cumsum(transects[[paste0('dpi', i-1)]]$Distance)
  
}

saveRDS(transects, file = paste0(save.dir, 'transects.rds'))
saveRDS(dpi, file = paste0(save.dir, 'dpi.rds'))

