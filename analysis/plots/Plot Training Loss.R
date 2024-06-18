library(jsonlite)

modelPath = '../../model/'
modelName = 'iota121v120240617_133013'
sidecar = jsonlite::read_json(paste0(modelPath, modelName, '.json'))
sidecar = jsonlite::read_json('../../model/iota121v1.json')

predictions = read.csv(paste0(modelPath, modelName, ' predictions.csv'), header = T)
colnames(predictions)[1] = 'true'
predictions$true = sidecar$labels[predictions$true+1]

p = rep(NA, nrow(predictions))

for (i in 1:length(p)) {
  p[i] = predictions[i,which(predictions$true[i] == colnames(predictions))]
}

loss = -log2(p)
hist(loss)
summary(loss)
mean(loss)
