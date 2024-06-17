library(data.table)

class.dir = '/data/new/analysis/camera1/classification/GAK_202207-REG-Gamma/'
segment.dir = '/data/new/analysis/camera1/segmentation/GAK_202207-REG/'
model.dir = '/data/new/training/training_set_20231002/data'

model.names = list.dirs(model.dir, full.names = F, recursive = F)
class.files = list.files(class.dir, pattern = '.csv')


thresholds = c(0.9, 0.7, 0)

for (i in 1:length(class.files)) {
  
  ## Load a classification file
  classifications = data.table::fread(paste0(class.dir, class.files[i]))
  colnames(classifications) = c('roi', model.names)
  classifications = classifications[classifications$roi != '',]
  
  ## Determine predicted category and confidence
  predictions = model.names[apply(classifications[,-1], 1, which.max)]
  prediction.conf = apply(classifications[,-1], 1, max)
  
  avi.name = gsub('_prediction.csv', '', class.files[i])
  
  ## Go through each category and move apprioriate ROIs
  for (name in model.names) {
    if (name %in% predictions) {
      
      ## Find ROIs that should be copied over and copy them.
      k = which(predictions == name & prediction.conf >= 0.9)
      if (length(k) > 0) {
        if (!dir.exists(paste0(class.dir, '/', name, '_0.9'))) {
          dir.create(paste0(class.dir, '/', name, '_0.9'))
        }
        file.copy(from = paste0(segment.dir, avi.name, '/', classifications$roi[k]),
                  to = paste0(class.dir, '/', name, '_0.9', '/', avi.name, '_', classifications$roi[k]))
      }
      
      k = which(predictions == name & prediction.conf >= 0.7 & prediction.conf < 0.9)
      if (length(k) > 0) {
        if (!dir.exists(paste0(class.dir, '/', name, '_0.7'))) {
          dir.create(paste0(class.dir, '/', name, '_0.7'))
        }
        file.copy(from = paste0(segment.dir, avi.name, '/', classifications$roi[k]),
                  to = paste0(class.dir, '/', name, '_0.7', '/', avi.name, '_', classifications$roi[k]))
      }
      
      k = which(predictions == name & prediction.conf < 0.7)
      if (length(k) > 0) {
        if (!dir.exists(paste0(class.dir, '/', name, '_0'))) {
          dir.create(paste0(class.dir, '/', name, '_0'))
        }
        file.copy(from = paste0(segment.dir, avi.name, '/', classifications$roi[k]),
                  to = paste0(class.dir, '/', name, '_0', '/', avi.name, '_', classifications$roi[k]))
      }
    }
  }
  message('Completed ', i, ' of ', length(class.files))
}


## Look up classification for a particular ROI:
roi.name = 'Camera1_VIPF-39-2022-07-21-23-59-18.521.avi_00408-00028.png'

tmp = strsplit(roi.name, split = '.avi_')[[1]]
k = grep(tmp[1], class.files)
classification = fread(paste0(class.dir, '/', class.files[k]))
colnames(classification) = c('roi', model.names)
l = which(classification$roi == tmp[2])
tmp = classification[l,-1]

t(round(tmp*100, 1))[tmp[1,] > 0.001,]
