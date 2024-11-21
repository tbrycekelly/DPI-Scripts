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

def Block(x, filters):
    x = tf.keras.layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def Model(input_shape, num_classes):
    """
    Deep convolutional network architecture
    Citation: Simonyan and Zisserman 2015. Very Deep Convolutional Networks for Large-Scale Image Recognition

    Configurations
    A: 64,      P,  128,        P,  256, 256,           P, 512, 512,            P,  512, 512
    B: 64, 64,  P,  128, 128,   P,  256, 256,           P, 512, 512,            P,  512, 512
    C: 64, 64,  P,  128, 128,   P,  256, 256, 256(1),   P, 512, 512, 512(1),    P,  512, 512, 512(1)
    D: 64, 64,  P,  128, 128,   P,  256, 256, 256,      P, 512, 512, 512,       P,  512, 512, 512
    E: 64, 64,  P,  128, 128,   P,  256, 256, 256, 256, P,  512, 512, 512, 512, P,  512, 512, 512, 512
    """
    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = augmentation_block(inputs)

    ## Configurations from 
    x = Block(x, 64)
    x = Block(x, 64)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Block(x, 128)
    x = Block(x, 128)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Block(x, 256)
    x = Block(x, 256)
    x = Block(x, 256)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Block(x, 512)
    x = Block(x, 512)
    x = Block(x, 512)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Block(x, 512)
    x = Block(x, 512)
    x = Block(x, 512)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Final layers
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model