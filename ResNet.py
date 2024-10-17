def Model(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = augmentation_block(inputs)
    
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # ResNet-34
    x = make_basic_block_layer(x, filter_num=64, blocks=3)
    x = make_basic_block_layer(x, filter_num=128, blocks=4, stride=2)
    x = make_basic_block_layer(x, filter_num=256, blocks=6, stride=2)
    x = make_basic_block_layer(x, filter_num=512, blocks=3, stride=2)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(img_input, x)
    return model


def make_basic_block_base(inputs, filter_num, stride=1):
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, kernel_initializer='he_normal', padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, kernel_initializer='he_normal', padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shortcut = inputs
    if stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride, kernel_initializer='he_normal')(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def make_basic_block_layer(inputs, filter_num, blocks, stride=1):
    
    x = make_basic_block_base(inputs, filter_num, stride=stride)

    for _ in range(1, blocks):
        x = make_basic_block_base(x, filter_num, stride=1)

    return x


## Functions for ResNet50, 101, and 152
def make_advanced_block_base(inputs, filter_num, stride=1):
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, kernel_initializer='he_normal', padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, kernel_initializer='he_normal', padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=filter_num*4, kernel_size=(1, 1), strides=1, kernel_initializer='he_normal', padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    

    shortcut = inputs
    if stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride, kernel_initializer='he_normal')(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def make_advanced_block_layer(inputs, filter_num, blocks, stride=1):
    
    x = make_advanced_block_base(inputs, filter_num, stride=stride)

    for _ in range(1, blocks):
        x = make_basic_block_base(x, filter_num, stride=1)

    return x