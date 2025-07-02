library(png)
library(batlow)

tileCommandFFV1 = function(input, output, n = 4) {
  paste0("ffmpeg -i '", input,
         "' -c:v ffv1 -level 3 -vf 'transpose=2,tile=", n,
         "x1' -pix_fmt gray -slices 16 -slicecrc 1 -y -threads 4 '", output, "'")
}

tileCommandAV1 =  function(input, output, n = 4, crf = 30) {
  paste0("ffmpeg -i '", input,
         "' -c:v libaom-av1 -crf ", crf, " -b:v 0 -vf 'transpose=2,format=gray,tile=", n,
         "x1' -cpu-used 8 -row-mt 1 -tiles 2x8 -y -threads 16 '", output, "'")
}

extractFrame = function(file, frame = 1, dest = NULL) {
  if (is.null(dest)) {
    dest = tempfile(fileext = '.png')
  }
  
  cmd = paste0('ffmpeg -i "', file, '" -vf "select=eq(n\\,', frame,')" -vframes 1 ', dest, ' -y')
  system(cmd, intern = T)
  
  res = readPNG(dest)
  res
}

generateComparison = function(file1, file2, output_image) {
  cmd = sprintf("ffmpeg -i %s -i %s -filter_complex \"
      [0:v]select=eq(n\\,0)[orig];
      [1:v]select=eq(n\\,0)[trans];
      [orig][trans]blend=all_mode=difference,format=gray,lut='val*2'[diff];
      [orig][trans][diff]hstack=inputs=3[out]
    \" -map \"[out]\" -frames:v 1 %s -y",
                shQuote(file1),
                shQuote(file2),
                shQuote(output_image)
  )
  
  system(cmd)
}

#### Setup
inputDir = '../../raw/camera0/test1/'
outputDirFFV1 = '../../raw/camera0/test1-ffv1/'
outputDirAV1 = '../../raw/camera0/test-av1/'
outputDirAVI = '../../raw/camera0/test-avi/'

if (!dir.exists(outputDirFFV1)) {
  dir.create(outputDirFFV1, recursive = T)
}
if (!dir.exists(outputDirAV1)) {
  dir.create(outputDirAV1, recursive = T)
}
if (!dir.exists(outputDirAVI)) {
  dir.create(outputDirAVI, recursive = T)
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
  system(cmd, intern = T)
  spotCheck[[output]] = extractFrame(file = output, frame = randomFrame)
  
  ## avi
  output = paste0(outputDirAVI, '/', videoFiles[i])
  cmd = tileCommandTile(input = paste0(inputDir, videoFiles[i]),
                        output = output,
                        n = 4)
  system(cmd, intern = T)
  spotCheck[[output]] = extractFrame(file = output, frame = randomFrame)
  
  
  ## Generate lossy transcode (AV1)
  output = paste0(outputDirAV1, '/', videoFiles[i])
  cmd = tileCommandAV1(input = paste0(inputDir, videoFiles[i]),
                       output = output,
                       n = 4,
                       crf = 20)
  system(cmd, intern = T)
  spotCheck[[output]] = extractFrame(file = output, frame = randomFrame)
}


x = c(1:800)
y = c(1:800)

image(spotCheck[[1]][x, y], col = rev(batlow(32)), main = names(spotCheck)[1])
image(spotCheck[[2]][x, y], col = rev(batlow(32)), main = names(spotCheck)[2])

image.default(spotCheck[[1]][x, y] - spotCheck[[2]][x, y], col = bam(33), zlim = c(-8/128,8/128))


sum(abs(spotCheck[[1]] - spotCheck[[2]]) > 3/128) / sum(spotCheck[[1]] > -1)
hist(128*abs(spotCheck[[1]] - spotCheck[[2]]), xlim = c(0,16), probability = T, breaks = c(-1:128)+0.5)


