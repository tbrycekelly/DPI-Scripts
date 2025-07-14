library(png)
library(batlow)
source('tileFunctions.R')

generateSidecar = function(file, camera, cruise, transect, aviPath) {
  
  md5Hash = tools::md5sum(file)
  metadata = checkFile(file)
  metadata$scanrate = eval(parse(text = metadata$frameRate))*metadata$width
  
  sidecar = list(
    filename = basename(file), # filename
    cruise = cruise,
    transect = transect,
    camera = camera,
    aviFilePath = aviPath,
    path = dirname(file), # original path (not useful except for retrospective)
    md5hash = md5Hash, # unique to the ffv1 encoding. /. 
    metadata = metadata, # information about the encoding. 
    sidecarCreation = list(
      datetime = Sys.time(),
      system = Sys.info()
    )
  )
  
  write_json(x = sidecar, path = gsub('.mkv', '.json', file))
}


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

## Convert each file into ffv1 and av1
for (i in 1:length(videoFiles)) {
  
  ## FFV1
  output = paste0(outputDirFFV1, '/', videoFiles[i])
  output = gsub('.avi', '.mkv', output)
  encodeFFv1(input = paste0(inputDir, videoFiles[i]), output = output)
  generateSidecar(output, )
  
  output = paste0(outputDirAV1, '/', videoFiles[i])
  output = gsub('.avi', '.mkv', output)
  encodeAV1(input = paste0(inputDir, videoFiles[i]), output = output, crf = 8)
  generateSidecar(output)
}

