source('processing/low level utilities.R')
source('processing/mid level utilities.R')

#### User input: 

trainingDir = '../../training/training_set_20240711/'

#### Autopilot from here:

## Load in list of all files in directory:
roiFiles = list.files(trainingDir, pattern = '.png', full.names = T, recursive = T)
roiFiles = gsub(trainingDir, '', roiFiles)
roiFiles = strsplit(roiFiles, split = '/')

roiIndex = data.frame(class = rep(NA, length(roiFiles)),
                      filename = NA)

for (i in 1:nrow(roiIndex)) {
  if (length(roiFiles[[i]]) == 2) {
    roiIndex$class[i] = roiFiles[[i]][1]
    tmp = strsplit(roiFiles[[i]][2], split = ' ')[[1]]
    roiIndex$filename[i] = tmp[2]
  }
}

message('Found ', length(roiFiles), ' image files across ', length(unique(roiIndex$class)), ' categories.')

classes = data.frame(class = unique(roiIndex$class), N = NA)

for (i in 1:nrow(classes)) {
  k = roiIndex$class == classes$class[i]
  classes$N[i] = sum(k)
  message('Found \t', sum(k), '\t entries for \t', classes$class[i], ' (', round(100 * sum(k) / nrow(roiIndex), 1), '%).')
}

classes = classes[order(classes$N),]
