library(png)
library(batlow)
library(SimpleGridder)
source('tileFunctions.R')

#### Setup
inputFile = 'Camera2_VIPF-256-2022-07-22-02-57-53.315.avi'
inputDir = '../../raw/camera0/test1/'
outputDir = '../../raw/camera0/test1-comparison/'

if (!dir.exists(outputDir)) {
  dir.create(outputDir, recursive = T)
}

spotCheck = list()

randomFrame = sample(1:200, 1)
  
## FFV1
output = paste0(outputDir, '/', inputFile)
output = gsub('.avi', '.mkv', output)
cmd = tileCommandFFV1(input = paste0(inputDir, inputFile),
                      output = output,
                      n = 4)
system(cmd, intern = T)
spotCheck[[output]] = extractFrame(file = output, frame = randomFrame)
  
## Generate lossy transcode (AV1)
CRF = c(6, 10, 16, 22, 28, 34)
for (crf in CRF) {
  output = paste0(outputDir, '/', inputFile)
  output = gsub('.avi', paste0('-', crf, '-av1.mkv'), output)
  cmd = tileCommandAV1(input = paste0(inputDir, inputFile),
                        output = output,
                        n = 4,
                        crf = crf)
  system(cmd, intern = T)
  spotCheck[[paste0('crf', crf)]] = extractFrame(file = output, frame = randomFrame)
}

par(plt = c(0,1,0,1))
image.default(spotCheck[[i]], col = greyscale(256), zlim = c(0,1))

x = c(400:600)
y = c(1000:1200)


for (i in 1:length(spotCheck)) {
  par(plt = c(0,1,0,1))
  image.default(spotCheck[[i]][x,y], col = greyscale(256), zlim = c(0,0.9))
}

par(plt = c(0,1,0,1))
image.default(spotCheck[[1]][x, y] - spotCheck[[2]][x, y], col = bam(15), zlim = c(-8/128,8/128))
colorbar(bam(15), zlim = c(-8/128,8/128))

x = c(1:800)
y = c(1:800)

image(spotCheck[[1]][x, y], col = rev(batlow(32)), main = names(spotCheck)[1])
image(spotCheck[[2]][x, y], col = rev(batlow(32)), main = names(spotCheck)[2])

for (i in seq(1, length(spotCheck), by = 2)) {
  image.default(spotCheck[[i]][x, y] - spotCheck[[i+1]][x, y], col = bam(33), zlim = c(-4/128,4/128))
}

image.default(spotCheck[[1]][x, y] - spotCheck[[26]][x, y], col = bam(33), zlim = c(-8/128,8/128))

sum(abs(spotCheck[[1]] - spotCheck[[2]]) > 3/128) / sum(spotCheck[[1]] > -1)
hist(128*abs(spotCheck[[1]] - spotCheck[[2]]), xlim = c(0,16), probability = T, breaks = c(-1:128)+0.5)


sourceDir = '/Volumes/shuttle/SKQ202513S/camera1/'
outputDir = '/Volumes/shuttle/SKQ202513S/camera1-ffv1/'

originalFiles = list.files(sourceDir, pattern = '.avi', full.names = T, recursive = T)

if (!dir.exists(outputDir)) {
  dir.create(outputDir, recursive = T)
}

for (i in 1:10) {
  tmp = gsub(sourceDir, outputDir, originalFiles[i])
  
  if (!dir.exists(dirname(tmp))) {
    dir.create(dirname(tmp))
  }
  cmd = tileCommandFFV1(input = originalFiles[i],
                        output = tmp)
  system(cmd)
}

