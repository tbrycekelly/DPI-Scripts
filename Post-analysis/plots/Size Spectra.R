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
