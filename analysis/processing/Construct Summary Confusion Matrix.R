
confusion = read.csv('../../model/thesis121v3 confusion.csv')
categories = confusion[,1]
confusion = confusion[,-1]

tmp = strsplit(categories, split = '_')
major = c()

for (i in 1:length(tmp)) {
  major = c(major, tmp[[i]][1])
}
maj = major
major = unique(major)

sumconfusion = array(NA, dim = rep(length(major),2))

for (i in 1:nrow(sumconfusion)) {
  for (j in 1:ncol(sumconfusion)) {
    l1 = grepl(major[i], maj)
    l2 = grepl(major[j], maj)
    
    sumconfusion[i,j] = sum(confusion[l1,l2])
  }
}
rownames(sumconfusion) = major
colnames(sumconfusion) = major

write.csv(sumconfusion, file = '../../model/thesis121v1 sumconfusion.csv')
