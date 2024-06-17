#' @export
export.morpho = function() {
  
  ## Remove and create export file
  message('To make the zip file: zip -FSr ../export.zip ./*')
  
  ## Copy to container
  message('to Copy data to container: sudo docker cp /media/plankline/Data/Data/osu_test/export.zip morphocluster2_morphocluster_1:/data')
  
  ## activate docker
  message('Log in: sudo docker-compose exec morphocluster bash')
  message('First: . ./activate; cd /data')
  
  ## Run feature extraction
  
} 