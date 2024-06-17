path = '../../analysis/camera0/classification/shadowgraph-REG-iota121v1/'
prediction.files = list.files(path, pattern = 'predictionlist.csv', full.names = T)

predictions = read.csv(prediction.files[1])
for (i in 2:length(prediction.files)) {
  tmp = read.csv(prediction.files[i])
  predictions = rbind(predictions, tmp)
}

counts = data.frame(category = unique(predictions$prediction),
           count = sapply(1:length(unique(predictions$prediction)), function(x) { sum(unique(predictions$prediction)[x] == predictions$prediction)}))
counts = counts[order(counts$count, decreasing = T),]

{
  par(plt = c(0.1, 0.98, 0.3, 0.98))
  plot(NULL,
       NULL,
       xlim = c(1,nrow(counts)),
       ylim = c(0, 4.3),
       xlab = '',
       ylab = 'Counts',
       xaxt = 'n',
       yaxt = 'n')
  
  axis(1, at = c(1:nrow(counts)), labels = counts$category, las = 2, cex.axis = 0.7)
  
  abline(h = c(0:4), col = 'grey')
  abline(h = log10(c(1:9)*1), col = 'grey', lty = 2)
  abline(h = log10(c(1:9)*10), col = 'grey', lty = 2)
  abline(h = log10(c(1:9)*100), col = 'grey', lty = 2)
  abline(h = log10(c(1:9)*1000), col = 'grey', lty = 2)
  abline(h = log10(c(1:9)*10000), col = 'grey', lty = 2)
  axis(2, at = c(0:4), labels = 10^c(0:4))
  
  points(log10(counts$count), pch = 15)
}


