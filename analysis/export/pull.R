#' @title Pull Classification Category
#' @description This function will extract images and generate an index file (morphocluster compatible) from a project directory.
#' @param project.dir The file path to the desired project directory
#' @param p The probability threshold for extracting classified images (0-1)
#' @param out.dir The output directory to build the file structure (default: ./tmp)
#' @param scratch The scratch directory to put temporary files (default: /tmp/pull)
#' @author Thomas Bryce Kelly
#' @export
pull = function(seg.dir, class.dir, out.dir, p = 0.5, scratch = '/tmp/pull') {
  
  ## Setup output folder for image files
  
  if (!dir.exists(scratch)) {
    message('Creating scartch directory.')
    dir.create(scratch)
  } else {
    stop('Scratch directory exists, must not exist for data protection! Dir = ', scratch)
  }
  
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
    
    message('Extracting TAR file to temporary directory...')
    tar.file = gsub('.csv', '.tar', basename(class.files[i]))
    tar.file = paste0(seg.dir, '/', tar.file)
    
    if (file.exists(tar.file)) {
      system(paste0('tar -xf "', tar.file, '" -C ', scratch, ' --strip-components=4 --wildcards "*.png"'))
    } else {
      stop('No matching Tar file found!')
    }
    
    paths = strsplit(data$image, '/')
    copied = rep(F, nrow(data))
    for (taxa in colnames(data)[-1]) {
      l = which(data[[taxa]] > p)
      
      if (length(l) > 0) {
        if (!dir.exists(paste0(out.dir, '/', taxa))) {
          dir.create(paste0(out.dir, '/', taxa))
        }
        copied[l] = T
        
        for (k in l) {
          count = count + 1
          file.copy(from = paste0(scratch, '/', paths[[k]][length(paths[[k]])]), to = paste0(out.dir, '/', taxa))
          #file.remove(paste0(scratch, '/', paths[[k]][length(paths[[k]])]))
        }
      }
    }
    
    ## Copy over whatever files are left (unclassified objects)
    if (!dir.exists(paste0(out.dir, '/_unsorted/'))) {dir.create(paste0(out.dir, '/_unsorted/'))}
    for (k in which(!copied)) {
      file.copy(from = paste0(scratch, '/', paths[[k]][length(paths[[k]])]), to = paste0(out.dir, '/_unsorted/'))
    }
    
    file.remove(list.files(scratch, pattern = '*', full.names = T))
  }
  
  file.remove(scratch)
  
  
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






