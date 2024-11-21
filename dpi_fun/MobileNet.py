import tensorflow as tf

def augmentation_block(x):
    x = tf.keras.layers.Rescaling(-1. / 255, 1)(x) # Invert shadowgraph image (white vs black)
    x = tf.keras.layers.RandomRotation(1, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomZoom(0.25, fill_value=0.0, fill_mode='constant')(x)
    x = tf.keras.layers.RandomTranslation(0.25, 0.25, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = tf.keras.layers.RandomBrightness(0.25, value_range=(0.0, 1.0))(x)
    x = tf.keras.layers.RandomContrast(0.25)(x)
    x = tf.keras.layers.GaussianNoise(0.1)(x)
    return x

def ConvDWBlock(x, filters, stride):
    x = tf.keras.layers.DepthwiseConv2D(filters, (3, 3), strides = stride, padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), strides = (1, 1), padding='same', kernel_initializer = tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def ConvBlock(x, filters):
    x = tf.keras.layers.Conv2D(filters, (1, 1), strides = (1, 1), padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Model(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = augmentation_block(inputs)

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
    
    model = tf.keras.models.Model(inputs, x)
    return model