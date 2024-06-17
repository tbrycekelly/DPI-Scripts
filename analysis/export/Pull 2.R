#' @title Pull Classification Category
#' @description This function will extract images and generate an index file (morphocluster compatible) from a project directory.
#' @param project.dir The file path to the desired project directory
#' @param p The probability threshold for extracting classified images (0-1)
#' @param out.dir The output directory to build the file structure (default: ./tmp)
#' @param scratch The scratch directory to put temporary files (default: /tmp/pull)
#' @author Thomas Bryce Kelly
#' @export
pull = function(seg.dir, class.dir, out.dir, p = 0) {
  
  ## Setup output folder for image files
  
  if (!dir.exists(out.dir)) {
    message('Creating output directory.')
    dir.create(out.dir)
  }
  
  count = 0
  a = Sys.time() # Timer
  
  ## Load each classification file, identify target files, extract if desired
  class.files = list.files(class.dir, pattern = '.csv', full.names = T)
  
  for (i in 1:length(class.files)) {
    message('Reading csv ', i, ' of ', length(class.files), '...')
    data = data.table::fread(class.files[i])
    data = as.data.frame(data)
    
    message('Extracting ZIP file...')
    zip.file = gsub('_prediction.csv', '.zip', basename(class.files[i]))
    zip.file = paste0(seg.dir, '/', zip.file)
    
    if (file.exists(zip.file)) {
      exdir = gsub('.zip', '', zip.file)
      unzip(zipfile = zip.file, exdir = gsub('.zip', '', zip.file))
    } else {
      stop('No matching ZIP file found!')
    }
    
    copied = rep(F, nrow(data))
    maxClass = apply(data[,-1], 1, which.max)
    
    for (i in 2:ncol(data)) {
      taxa = colnames(data)[i]
      l = which(maxClass == i & data[,i] >= p)
      
      if (length(l) > 0) {
        if (!dir.exists(paste0(out.dir, '/', taxa))) {
          dir.create(paste0(out.dir, '/', taxa))
        }
        copied[l] = T
        
        for (k in l) {
          count = count + 1
          file.copy(from = paste0(exdir, '/', data[l[k],1]), to = paste0(out.dir, '/', taxa))
          file.remove(paste0(exdir, '/', data[l[k],1]))
        }
      }
    }
    
    ## Copy over whatever files are left (unclassified objects)
    if (!dir.exists(paste0(out.dir, '/_unsorted/'))) {dir.create(paste0(out.dir, '/_unsorted/'))}
    for (k in which(!copied)) {
      file.copy(from = paste0(exdir, '/', data[k,1]), to = paste0(out.dir, '/_unsorted/'))
    }
    
    file.remove(list.files(exdir, pattern = '*', full.names = T))
  }
  
  file.remove(exdir)
  
  
  ## Make morphocluster index
  #images = list.files(path = out.dir, recursive = T, pattern = '*.png', full.names = F)
  #name = strsplit(images, split = '/')
  #index = data.frame(object_id = NA, path = images)
  
  #for (i in 1:nrow(index)) {
  #  index$object_id[i] = gsub('.png', '', name[[i]][length(name[[i]])])
  #}
  
  #utils::write.csv(index, file = paste0(out.dir, '/index.csv'), row.names = F)
  
  
  ## Done
  message('Found ', count, ' valid files (in ', round(as.numeric(difftime(Sys.time(), a, units = 'secs'))), ' seconds).')
}



pull(seg.dir = '../../../analysis/camera0/segmentation/test1-REG/',
     class.dir = '../../../analysis/camera0/classification/test1-REG-iota121v1-clusters/',
     out.dir = '../../../analysis/camera0/pull')


