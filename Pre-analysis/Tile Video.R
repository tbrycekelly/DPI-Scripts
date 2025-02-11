tileCommand =  function(input, output, n = 4) {
  paste0("ffmpeg -i '", input, "' -c:v ffv1 -level 3 -vf 'tile=1x", n, "' -pix_fmt gray -slices 16 -slicecrc 1 -threads 16 '", output, "'")
}

inputDir = '../../raw/camera0/test1/'
outputDir = '../../raw/camera0/testTile'


dir.create(outputDir, recursive = T)

videoFiles = list.files(inputDir, pattern = '.avi')

for (i in 1:length(videoFiles)) {
  cmd = tileCommand(input = paste0(inputDir, videoFiles[i]),
                    output = gsub('avi', 'mkv', paste0(outputDir, '/', videoFiles[i])),
                    n = 4)
  message(cmd)
  #system(cmd, intern = T)
}
