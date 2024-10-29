library(openxlsx)
library(TheSource)
library(PlanklinePS)


## Setup
root.dir = '/media/plankline/Data/Sensor/SKQJ2022_data/'

## Default directories
directories = list(
  root = root.dir,
  gps = paste0(root.dir, 'VIPF_LOGS/GPS Publisher/'),
  ctd1 = paste0(root.dir, 'VIPF_LOGS/CTD 1 Publisher/'),
  ctd2 = paste0(root.dir, 'VIPF_LOGS/CTD 2 Publisher/'),
  ad = paste0(root.dir, 'VIPF_LOGS/AD1216 Publisher/'),
  fl1 = paste0(root.dir, 'VIPF_LOGS/Fluorometer 1 Publisher/'),
  fl2 = paste0(root.dir, 'VIPF_LOGS/Fluorometer 2 Publisher/'),
  eng = paste0(root.dir, 'VIPF_LOGS/Inclinometer Publisher/'),
  lisst = paste0(root.dir, 'LISST/'),
  acs = paste0(root.dir, 'ACS/'),
  suna = paste0(root.dir, 'SUNA/'),
  save = paste0(root.dir, '/_rdata/')
)

if (!dir.exists(directories$root)) {stop('Root directory not found. Aborting!')}

for (i in 2:length(directories)) {
  if (!dir.exists(directories[[i]])) {
    message('Creating missing directory: ', directories[[i]])
    dir.create(directories[[i]])
  }
}

### GPS Files
gps.files = list.files(path = directories$gps,
                       full.names = T,
                       recursive = T,
                       pattern = '*.csv')
gps = load.dpi.files(gps.files)

colnames(gps) = c('Time', 'Message', 'datetime', 'Latitude', 'Hemisphere',
                  'Longitude', 'other hemi',
                  'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14','v15','vq6','v17','v18')

## Fix structure and pull out good data only
gps$Latitude = floor(gps$Latitude/100) + (gps$Latitude - floor(gps$Latitude/100)*100)/60
gps$Longitude = -1 * (floor(gps$Longitude/100) + (gps$Longitude - floor(gps$Longitude/100)*100)/60)
gps = gps[,c(1,4,6)]

gps = gps[gps$Time > make.time(2000),]

saveRDS(gps, file = paste0(directories$save, 'gps.rds'))



#### CTD 1
## Initial setup
ctd.files = list.files(directories$ctd1,
                       full.names = T,
                       pattern = '*.csv',
                       recursive = T)

ctd = load.dpi.files(ctd.files)

colnames(ctd) = c('Time', 'Temperature', 'Conductivity', 'Pressure', 'Depth', 'Salinity', 'Sound.Velocity')


## Calculate
ctd$SigmaTheta = calc.sigma.theta(S = ctd$Salinity, Tmp = ctd$Temperature, P = ctd$Pressure, verbose = F)

ctd = ctd[ctd$Time > make.time(2000),]
saveRDS(ctd, file = paste0(directories$save, 'ctd1.rds'))


#### CTD 2
## Initial setup
ctd.files = list.files(directories$ctd2,
                       full.names = T,
                       pattern = '*.csv',
                       recursive = T)

ctd = load.dpi.files(ctd.files)
ctd = ctd[ctd$Time > make.time(2000),]

colnames(ctd) = c('Time', 'Temperature2', 'Conductivity2', 'Pressure2', 'Depth2', 'Salinity2', 'Sound.Velocity2')

## Calculate
ctd$SigmaTheta2 = calc.sigma.theta(S = ctd$Salinity2, Tmp = ctd$Temperature2, P = ctd$Pressure2, verbose = F)

saveRDS(ctd, file = paste0(directories$save, 'ctd2.rds'))



#### Analog
## Set up format
ad.files = list.files(directories$ad,
                      full.names = T,
                      pattern = '.csv',
                      recursive = T)

ad = load.dpi.files(ad.files)

colnames(ad) = c('Time', 'Oxygen', 'PAR', 'pH', 'REDOX', 'Checksum', 'checksum2')

ad = ad[ad$Time > make.time(2000),]
saveRDS(ad, file = paste0(directories$save, 'analog.rds'))


#### Fluorometer 1
fl.files = list.files(directories$fl1,
                      full.names = T,
                      recursive = T, 
                      pattern = '*.csv')
fl = load.dpi.files(fl.files)

colnames(fl) = c('Time', 'Chl.wave', 'Chl', 'FDOm.wave', 'FDOM', 'Phycocyanin.wave', 'Phycocyanin')

## Filter
fl$Chl[fl$Chl > 4e3 | fl$Chl < 10] = NA
fl$FDOM[fl$FDOM > 4e3 | fl$FDOM < 10] = NA
fl$Phycocyanin[fl$Phycocyanin > 4e3 | fl$Phycocyanin < 10] = NA

## Convert counts to real units
# Calibration from ???
fl$Chl = (fl$Chl - 49) * 0.0121
fl$FDOM = (fl$FDOM - 50) * 0.0914
fl$Phycocyanin = (fl$Phycocyanin - 46) * 0.0422

fl = fl[fl$Time > make.time(2000),]
saveRDS(fl, file = paste0(directories$save, 'fluorometer1.rds'))


#### Fluorometer 2
fl.files = list.files(directories$fl2,
                      full.names = T,
                      recursive = T,
                      pattern = '*.csv')
fl = load.dpi.files(fl.files)

colnames(fl) = c('Time', 'S470.wave', 'S470', 'S532.wave', 'S532', 'S650.wave', 'S650')

## Filter
fl$S470[fl$S470 > 4e3 | fl$S470 < 10] = NA
fl$S532[fl$S532 > 4e3 | fl$S532 < 10] = NA
fl$S650[fl$S650 > 4e3 | fl$S650 < 10] = NA

## Convert counts to real units
# Calibration from ???
fl$S470 = (fl$S470 - 35) * 0.00001181
fl$S532 = (fl$S532 - 50) * 0.000008216
fl$S650 = (fl$S650 - 50) * 0.00000411

fl = fl[fl$Time > make.time(2000),]
saveRDS(fl, file = paste0(directories$save, 'fluorometer2.rds'))



#### Eng
eng.files = list.files(directories$eng,
                       full.names = T,
                       pattern = '*.csv',
                       recursive = T)

eng = load.dpi.files(eng.files)

colnames(eng) = c('Time', 'AccelerationX', 'AccelerationY', 'AccelerationZ', 'Roll', 'Pitch', 'T')
eng = eng[eng$Time > make.time(2000),]
saveRDS(eng, file = paste0(directories$save, 'engineering.rds'))



#### LISST
# TODO: timestamps are not right!
lisst.files = list.files(directories$lisst,
                         full.names = T,
                         pattern = '*_rs.csv',
                         recursive = T)
lisst = load.lisst.files(lisst.files)

saveRDS(lisst, file = paste0(directories$save, 'lisst.rds'))



#### ACs
acs.files = list.files(directories$acs,
                       full.names = T,
                       pattern = '*.dat',
                       recursive = T)
acs = load.acs.files(acs.files)

acs = cbind(Time = acs$Time, acs[,which(colnames(acs) != 'Time')])
acs$V178 = NULL

saveRDS(acs, file = paste0(directories$save, 'acs.rds'))


#### SUNA
suna.files = list.files(directories$suna,
                       full.names = T,
                       pattern = '*.sbslog',
                       recursive = T)
suna = load.suna.files(suna.files, skip = 15, min.size = 250000)
suna = suna[suna$Time > make.time(2022),]
suna$Time = suna$Time - 86400

saveRDS(suna, file = paste0(directories$save, 'suna.rds'))
