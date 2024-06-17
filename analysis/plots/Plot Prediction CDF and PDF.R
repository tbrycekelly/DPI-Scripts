path = '../../analysis/camera0/classification/shadowgraph-REG-iota121v1/'
prediction.files = list.files(path, pattern = 'prediction.csv', full.names = T)

predictions = read.csv(prediction.files[1])
for (i in 2:length(prediction.files)) {
  tmp = read.csv(prediction.files[i])
  predictions = rbind(predictions, tmp)
}

{
  tmp = apply(predictions[,-1], 1, max)
  
  par(mfrow = c(2,1), plt = c(0.15,0.98, 0.15, 0.96))
  dense = density(tmp)
  plot(dense$x,
       dense$y,
       type = 'l',
       xaxs = 'i',
       xlim = c(0, 1),
       yaxs = 'i',
       ylim = c(0, max(pretty(dense$y))),
       xlab = '',
       ylab = 'PDF')
  grid()
  
  plot(ecdf(tmp), yaxs = 'i', xaxs = 'i', xlim = c(0,1),
       ylab = 'CDF')
  grid()
}


{
  tmp = apply(predictions[,-1], 1, max)
  for (i in 2:ncol(predictions)) {
    name = names(predictions)[i]
    cat = apply(predictions[,-1], 1, which.max)
    
    k = cat == i
    par(mfrow = c(2,1), plt = c(0.15,0.98, 0.15, 0.96))
    dense = density(tmp[k])
    plot(dense$x,
         dense$y,
         type = 'l',
         xaxs = 'i',
         xlim = c(0, 1),
         yaxs = 'i',
         ylim = c(0, max(pretty(dense$y))),
         xlab = '',
         ylab = 'PDF')
    mtext(name, line = -1.5, adj = 0.02)
    grid()
    
    plot(ecdf(tmp[k]), yaxs = 'i', xaxs = 'i', xlim = c(0,1),
         ylab = 'CDF')
    
    mtext(name, line = -1.5, adj = 0.02)
    grid()
  }
}

