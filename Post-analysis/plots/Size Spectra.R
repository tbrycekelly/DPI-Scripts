sizeFile = '../../analysis/camera0/segmentation/test5-REG/deploy1 statistics.csv'
sizeFileRT = '~/Google Drive/Shared drives/DPI/Shadowgraph/NearPost-Comparison/psd_resBay_deploy1.csv'

sizes = read.csv(sizeFile)
sizes = sizes[sizes$w * sizes$h > 5,]
sizesRT = read.csv(sizeFileRT)

cuts = seq(0, 20000, by = 100)
hist(sizes$perimeter, breaks = cuts, xlim = c(0, 2e3), xlab = 'Perimeter (pixel)')
tmp = hist.default(sizesRT$perimeter/5, breaks = cuts, xlim = c(0, 2e3), plot = F)

for (i in 1:length(tmp$counts)) {
  rect(tmp$breaks[i], ybottom = 0, xright = tmp$breaks[i+1], ytop = tmp$counts[i], col = '#dd000040')
}


plot(sizes$perimeter, sizes$perimeter^2 / sizes$area, xlim = c(0, 200), ylim = c(0, 300))




#### From training set
loadCSV = function(path) {
  tmp = read.csv(path)
  for (f in unique(tmp$frame)) {
    if (sum(tmp$frame == f) > 1) {
      k = which(tmp$frame == f)
      j = which.max(tmp$area[k])
      tmp = tmp[-c(k[-c(j)]),]
      #message('Removed duplicate at frame ', f, ' (', nrow(tmp), ')' )
    }
  }
  tmp
}


resultFiles = list.files('../../training/training_set_20241001-REG/', pattern = '.csv', full.names = T)
weights = array(0, dim = c(length(resultFiles), length(breaks)-1))

pdf('../../export/Size Spectra.pdf')
par(mfrow = c(2,2))
for (i in 1:length(resultFiles)) {
  tmp = loadCSV(resultFiles[i])
  
  
  name = strsplit(resultFiles[i], '/')[[1]]
  name = gsub(' statistics.csv', '', name[length(name)])
  
  breaks = seq(0, 1e4, by = 100)
  hist(tmp$perimeter,
       breaks = breaks,
       xaxs = 'i',
       yaxs = 'i',
       freq = F,
       xlim = c(0, 1000),
       ylim = c(0, 0.01),
       xlab = 'Perimeter (pixel)',
       main = name)
  
  lines(counts$mids, avg/100, lwd = 2)
  mtext(paste0('n = ', sum(counts$counts)), side = 3, line = -2, adj = 0.9)
  
  counts = hist(tmp$perimeter,
               breaks = breaks,
               xaxs = 'i',
               yaxs = 'i',
               plot = F)
  
  weights[i,] = counts$counts / sum(counts$counts)
  
  plot(counts$mids,
       log10(weights[i,] / avg),
       ylim = c(-2, 2),
       yaxt = 'n',
       xlim = c(0, 1000),
       xlab = 'Perimeter (pixels)',
       ylab = 'Rel to Avg',
       xaxs = 'i')
  
  axis(2, at = c(-2:2), labels = 10^(-2:2))
  axis(2, at = log10(10*(1:9)), labels = F)
  axis(2, at = log10(1:9), labels = F)
  axis(2, at = log10(0.1*1:9), labels = F)
  axis(2, at = log10(0.01*1:9), labels = F)
  abline(h = c(-2:2), col = 'black', lty = 3)
  abline(h = 0, lty = 2)
  
}
dev.off()

avg = apply(weights, 2, median)




