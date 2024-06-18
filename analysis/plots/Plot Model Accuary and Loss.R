
model.path = '~/Downloads/iota201v1 (1).log'

log = read.csv(gsub('.keras', '.log', model.path))

{
  par(mfrow = c(2,1), plt = c(0.15, 0.95, 0.25, 0.95))
  plot(log$epoch,
       log$accuracy,
       ylim = c(0.85, 1),
       type = 'l',
       lwd = 2,
       xaxs = 'i',
       yaxs = 'i',
       ylab = 'Accuracy (%)',
       xlab = 'Epoch',
       xlim = range(pretty(log$epoch)))
  lines(log$epoch, log$val_accuracy, lwd = 2, col = 'blue')
  lines(runmed(log$epoch,15), runmed(log$val_accuracy, 15), lwd = 2, col = 'black')
  
  grid()c
  
  
  plot(log$epoch,
       log$loss,
       ylim = c(0, 1),
       type = 'l',
       lwd = 2,
       xaxs = 'i',
       yaxs = 'i',
       ylab = 'Loss',
       xlab = 'Epoch',
       xlim = range(pretty(log$epoch)))
  lines(log$epoch, log$val_loss, lwd = 2, col = 'blue')
  lines(runmed(log$epoch,15), runmed(log$val_loss, 15), lwd = 2, col = 'black')
  
  grid()
}
