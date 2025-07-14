library(jpeg)
library(png)
library(imager)

imgPath = '../../raw/camera0/test6/deploy1/sg006 20240513-141421-001.jpg'
outpath = tempfile(fileext = '.png')



{ ## Flatfield correction like linescan
  image = readJPEG(imgPath)
  image = image[,,1]
  
  center = dim(image)/2
  radius = 1150
  meangray = median(image[(center[1] - radius/2):(center[1] + radius/2), (center[2] - radius/2):(center[2] + radius/2)])
  
  for (i in 1:dim(image)[1]) {
    for (j in 1:dim(image)[2]) {
      if ((center[1] - i)^2 + (center[2] - j)^2 > radius^2) {
        image[i,j] = meangray
      }
    }
  }
  
  ## Quantile apporach
  field = apply(image, 1, function(x) {quantile(x, 0.975)})
  
  image = image / field
  image[image < 0] = 0
  image[image > 1] = 1
  
  png::writePNG(image, target = outpath)
  browseURL(outpath)
}


{ ## Median filters to enhance contrast?
  image = readJPEG(imgPath)
  image = image[,,1]
  
  center = dim(image)/2
  radius = 1150
  meangray = median(image[(center[1] - radius/2):(center[1] + radius/2), (center[2] - radius/2):(center[2] + radius/2)])
  
  for (i in 1:dim(image)[1]) {
    for (j in 1:dim(image)[2]) {
      if ((center[1] - i)^2 + (center[2] - j)^2 > radius^2) {
        image[i,j] = meangray
      }
    }
  }
  
  blurred = imager::isoblur(image, 1)
  
  image = image / field
  image[image < 0] = 0
  image[image > 1] = 1
  
  png::writePNG(image, target = outpath)
  browseURL(outpath)
}