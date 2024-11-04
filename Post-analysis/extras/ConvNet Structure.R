


ffv1Transcode = function(inName, outName) {
  ffv1String = paste0("ffmpeg -y -i ", inName, " -c:v ffv1 -vf 'format=gray' -vf 'tile=1x3' -level 3 ", outName, ".mkv")
  
  system(ffv1String)
}

ffv1Transcode('original.avi', 'ffv1')






h265Transcode = function(inName, outName, crf = 28) {
  h265String = paste0("ffmpeg -y -i ", inName, " -c:v libx265 -vf 'format=gray' -vf 'tile=1x8' -crf ", crf," ", outName, ".mkv")
  
  system(h265String)
}

h265Transcode('original.avi', 'h265_28')
h265Transcode('original.avi', 'h265_18', 18)
h265Transcode('original.avi', 'h265_22', 22)
h265Transcode('original.avi', 'h265_12', 12)






"
    x = Block(x, 64)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Block(x, 128)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Block(x, 256)
    x = Block(x, 256)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Block(x, 512)
    x = Block(x, 512)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Block(x, 512)
    x = Block(x, 512)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
"


par(plt = c(0,1,0,1))
plot(NULL, NULL, xlim = c(0,800), ylim = c(0,800), frame = F, xaxt = 'n', yaxt = 'n', xaxs = 'i', yaxs = 'i')

start = c(130, 650)

## 1 Block
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

## Pool
start = start + c(8, -10) * 1.5
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#ff7777', border = T)

# 1 Block
start = start + c(8, -10) * 1.5
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

## Pool
start = start + c(8, -10) * 1.5
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#ff7777', border = T)

# 2 Blocks
start = start + c(8, -10) * 1.5
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

## Pool
start = start + c(8, -10) * 1.5
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#ff7777', border = T)

# 2 Blocks
start = start + c(8, -10) * 1.5
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

## Pool
start = start + c(8, -10) * 1.5
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#ff7777', border = T)

# 2 Blocks
start = start + c(8, -10) * 1.5
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)

start = start + c(8, -10)
rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)


points(seq(start[1] - 175, start[1] + 175, length.out = 4096), rep(start[2] - 20, 4096), pch = '.')
points(seq(start[1] - 175, start[1] + 175, length.out = 4096), rep(start[2] - 30, 4096), pch = '.')
points(seq(start[1] - 175, start[1] + 175, length.out = 1000), rep(start[2] - 40, 1000), pch = '.')
points(seq(start[1] - 175, start[1] + 175, length.out = 63), rep(start[2] - 50, 63), pch = '.')
