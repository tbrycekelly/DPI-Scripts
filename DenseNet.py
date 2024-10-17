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

def conv_block(x, growth_rate):
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(growth_rate, (3, 3), padding = 'same', kernel_initializer = tf.keras.initializers.HeNormal())(x1)
    x = tf.keras.layers.concatenate([x, x1], axis = -1)
    return x


def dense_block(x, num_layers, growth_rate):
    for i in range(num_layers):
        x = conv_block(x, growth_rate)
    return x


def transition_block(x, compression):
    num_filters = int(x.shape[-1] * compression)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, (1, 1))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x


def Model(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = augmentation_block(inputs)

    # Initial convolution layer
    x = tf.keras.layers.Conv2D(256, (7, 7), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    ## DenseNet121 (112 internal)
    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=24, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=16, growth_rate=32)

    # Final layers
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model
