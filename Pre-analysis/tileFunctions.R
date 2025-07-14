library(jsonlite)

readMetadata = function(input) {
  res = list()
  for (i in 1:length(input)) {
    cmd = paste0('ffprobe -v quiet -print_format json -show_format -show_streams "', input[i], '"')
    metadata = system(cmd, intern = T)
    res[[basename(input[i])]] = jsonlite::fromJSON(metadata)
  }
  
  res
}




checkFileAVI = function(input) {
  
  meta = data.frame(file = basename(input), path = input, frameCount = NA, size = NA, width = NA, height = NA, pixelFormat = NA, frameRate = NA, bitrate = NA)
  for (i in 1:length(input)) {
    if (file.exists(input[i])) {
      tmp = readMetadata(input)
      meta$frameCount[i] = as.numeric(tmp[[1]]$streams$nb_frames)
      meta$width[i] = as.numeric(tmp[[1]]$streams$width)
      meta$height[i] = as.numeric(tmp[[1]]$streams$height)
      meta$frameRate[i] = tmp[[1]]$streams$r_frame_rate
      meta$pixelFormat[i] = tmp[[1]]$streams$pix_fmt
      meta$size[i] = as.numeric(tmp[[1]]$format$size)
      meta$bitrate[i] = as.numeric(tmp[[1]]$format$bit_rate)
    }
  }
  
  meta
}

checkFileFFv1 = function(input) {
  
  meta = data.frame(file = basename(input), path = input, frameCount = NA, size = NA, width = NA, height = NA, pixelFormat = NA, frameRate = NA, bitrate = NA)
  for (i in 1:length(input)) {
    if (file.exists(input[i])) {
      tmp = readMetadata(input)
      meta$frameCount[i] = as.numeric(tmp[[1]]$streams$nb_frames)
      meta$width[i] = as.numeric(tmp[[1]]$streams$width)
      meta$height[i] = as.numeric(tmp[[1]]$streams$height)
      meta$frameRate[i] = tmp[[1]]$streams$r_frame_rate
      meta$pixelFormat[i] = tmp[[1]]$streams$pix_fmt
      meta$size[i] = as.numeric(tmp[[1]]$format$size)
      meta$bitrate[i] = as.numeric(tmp[[1]]$format$bit_rate)
    }
  }
  
  meta
}


videoConcatenate = function(input, output) {
  fileList = 'filelist.txt'
  
  writeLines(paste0("file '", input, "'"), con = fileList)
  cmd = paste0("ffmpeg -f concat -safe 0 -i ", fileList," -c copy '", output, "'")
  system(cmd)
}

## Tile and encode commands
tileCommandFFv1 = function(input, output, n = 4) {
  paste0("ffmpeg -i '", input,
         "' -c:v ffv1 -level 3 -vf 'transpose=2,tile=", n,
         "x1' -pix_fmt gray -slices 16 -slicecrc 1 -y -threads 4 '", output, "'")
}

tileCommandAV1 =  function(input, output, n = 4, crf = 26) {
  paste0("ffmpeg -i '", input,
         "' -c:v libaom-av1 -crf ", crf, " -b:v 0 -vf 'transpose=2,format=gray,tile=", n,
         "x1' -cpu-used 8 -row-mt 1 -tiles 2x8 -y -threads 16 '", output, "'")
}

## helper functions
tileFFv1 = function(input, output, n = 4) {
  cmd = tileCommandFFv1(input, output, n)
  system(cmd)
}

tileAV1 = function(input, output, n = 4, crf = 10) {
  cmd = tileCommandAV1(input, output, n, crf)
  system(cmd)
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


## Tile and encode commands
encodeFFv1 = function(input, output) {
  cmd = paste0("ffmpeg -i '", input, "' -c:v ffv1 -level 3 -pix_fmt gray -slices 16 -slicecrc 1 -y -threads 4 '", output, "'")
  system(cmd)
}

encodeAV1 =  function(input, output, crf = 26) {
  cmd = paste0("ffmpeg -i '", input, "' -c:v libaom-av1 -crf ", crf, " -b:v 0 -cpu-used 8 -row-mt 1 -tiles 2x8 -y -threads 16 '", output, "'")
  system(cmd)
}



## 
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



