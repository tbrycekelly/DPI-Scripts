"
    # Conv 1
    x = tf.keras.layers.Conv2D(32, (3, 3), strides = (2, 2), padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = ConvDWBlock(x, 32, 1)
    x = ConvBlock(x, 64)
    
    x = ConvDWBlock(x, 64, 2)
    x = ConvBlock(x, 128)
    
    x = ConvDWBlock(x, 128, 1)
    x = ConvBlock(x, 128)
    
    x = ConvDWBlock(x, 128, 2)
    x = ConvBlock(x, 128)
    
    x = ConvDWBlock(x, 256, 1)
    x = ConvBlock(x, 256)
    
    x = ConvDWBlock(x, 256, 2)
    x = ConvBlock(x, 512)

    # x 5
    x = ConvDWBlock(x, 512, 1)
    x = ConvBlock(x, 512)
    
    x = ConvDWBlock(x, 512, 1)
    x = ConvBlock(x, 512)
    
    x = ConvDWBlock(x, 512, 1)
    x = ConvBlock(x, 512)
    
    x = ConvDWBlock(x, 512, 1)
    x = ConvBlock(x, 512)
    
    x = ConvDWBlock(x, 512, 1)
    x = ConvBlock(x, 512)
    
    x = ConvDWBlock(x, 512, 2)
    x = ConvBlock(x, 1024)
    
    x = ConvDWBlock(x, 1024, 2)
    x = ConvBlock(x, 1024)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Final layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
"

par(plt = c(0,1,0,1))
plot(NULL, NULL, xlim = c(0,800), ylim = c(0,800), frame = F, xaxt = 'n', yaxt = 'n', xaxs = 'i', yaxs = 'i')

start = c(130, 650)

rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)
start = start + c(5, -10)

for (i in 1:13) {
  ## DW Block 1
  rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#77aa77', border = T)
  start = start + c(5, -10) * 0.5
  rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)
  
  ## conv block 1
  start = start + c(5, -10) * 1.6
  rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#777777', border = T)
  start = start + c(9, -10) * 1.6
}

rect(start[1], start[2], start[1] + 40, start[2] + 40, col = '#ff7777', border = T)

points(seq(start[1] - 175, start[1] + 175, length.out = 1000), rep(start[2] - 30, 1000), pch = '.')
points(seq(start[1] - 175, start[1] + 175, length.out = 63), rep(start[2] - 40, 63), pch = '.')
