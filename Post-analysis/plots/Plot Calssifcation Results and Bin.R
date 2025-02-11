results = do.call(rbind, results)

k = which(results$pmax > 0.9)
results = results[k,]

tmp = strsplit(results$filename, split = ' ')

results$datetime = NA
for (i in 1:nrow(results)) {
  tic = tmp[[i]][3]
  results$datetime[i] = paste0(substr(tic, 1, 4), '-', substr(tic, 5, 6), '-', substr(tic, 7, 8), ' ',
                               substr(tic, 10, 11), ':', substr(tic, 12, 13), ':', substr(tic, 14, 15))
}

results$datetime = as.POSIXct(results$datetime, tz = 'UTC')
results = results[order(results$datetime),]
plot(diff(as.numeric(results$datetime)), pch = '.', ylim = c(0,60))


binnedResults = data.frame(datetime = unique(results$datetime))

for (org in unique(results$class)) {
  binnedResults[[org]] = NA
}

for (i in 1:nrow(binnedResults)) {
  k = which(results$datetime == binnedResults$datetime[i])
  for (org in colnames(binnedResults)[-1]) {
    binnedResults[i, org] = sum(results$class[k] == org)
  }
}

plot(binnedResults$datetime, binnedResults$copepod_poecilostomatoid, type = 'l')


binnedResults$copepod = apply(binnedResults[,grepl('copepod', colnames(binnedResults))], 1, sum)
plot(binnedResults$datetime, binnedResults$copepod, type = 'l')

binnedResults$particles = apply(binnedResults[,-1], 1, sum) - binnedResults$copepod
plot(binnedResults$datetime, binnedResults$particles, type = 'l')

openxlsx::write.xlsx(binnedResults, '../../export/shadowgraph2/Binned Results (v2).xlsx')

