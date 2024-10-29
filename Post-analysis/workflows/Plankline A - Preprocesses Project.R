library(PlanklinePS)


#### Project preparation and parameter setting
## Setup
base.dir = '/media/plankline/Data/Data/SNR10/'
frame.rate = 20.5
bin.width = 1 # second(s)
p.threshold = 0.5 # probability threshold


out.dir = paste0(base.dir, '/R/')
if (!dir.exists(out.dir)) { dir.create(out.dir)}
    
process.classifications(dir = base.dir)
    
#process.measurements(base.dir = base.dir,
#                     out.dir = out.dir,
#                     frame.rate = frame.rate)
    
generate.bin(dir = base.dir,
             p.threshold = p.threshold,
             frame.rate = frame.rate,
             bin.width = bin.width)
