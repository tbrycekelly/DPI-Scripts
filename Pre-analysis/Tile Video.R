library(png)
library(batlow)
source('tileFunctions.R')

#### Setup
inputDir = '../../raw/camera0/test1/'
outputDirFFV1 = '../../raw/camera0/test1-ffv1/'
outputDirAV1 = '../../raw/camera0/test-av1/'

if (!dir.exists(outputDirFFV1)) {
  dir.create(outputDirFFV1, recursive = T)
}
if (!dir.exists(outputDirAV1)) {
  dir.create(outputDirAV1, recursive = T)
}

videoFiles = list.files(inputDir, pattern = '.avi')
spotCheck = list()

## Convert each file into ffv1 and tile (1x4). Then generate lossy AV1 as well.
for (i in 1:length(videoFiles)) {
  randomFrame = sample(1:200, 1)
  
  ## FFV1
  output = paste0(outputDirFFV1, '/', videoFiles[i])
  output = gsub('.avi', '.mkv', output)
  cmd = tileCommandFFV1(input = paste0(inputDir, videoFiles[i]),
                    output = output,
                    n = 4)
  #system(cmd, intern = T)
  #spotCheck[[output]] = extractFrame(file = output, frame = randomFrame)
  
  ## Generate lossy transcode (AV1)
  output = paste0(outputDirAV1, '/', videoFiles[i])
  output = gsub('.avi', '.mkv', output)
  cmd = tileCommandAV1(input = paste0(inputDir, videoFiles[i]),
                       output = output,
                       n = 4,
                       crf = 10)
  system(cmd, intern = T)
  #spotCheck[[output]] = extractFrame(file = output, frame = randomFrame)
}


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


cmd = videoConcatenate(paste0(inputDir, videoFiles), paste0(outputDir, 'combined.avi'))
system(cmd)
cmd = tileCommandFFV1(paste0(outputDir, 'combined.avi'), paste0(outputDir, 'combined-ffv1.avi'), n = 4)
system(cmd)
file.remove(paste0(outputDir, 'combined.avi'))
