#!/usr/bin/env python3
"""DenseNet training script for DPI-Scripts

Usage:
    ./train.py

License:
    MIT License

    Copyright (c) 2023 Thomas Kelly

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import os
import shutil
import argparse
import logging
import logging.config
import configparser
import tqdm
from time import time
import psutil
from multiprocessing import Pool
import datetime
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from PIL import Image
import pandas as pd
import json
import platform
import time
import csv
import sys

## Source import.py
exec(open("./imports.py").read())

def conv_block(x, growth_rate):
    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(growth_rate, (3, 3), padding='same')(x1)
    x = tf.keras.layers.concatenate([x, x1], axis=-1)
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

def augmentation_block(x):
    x = tf.keras.layers.Rescaling(-1. / 255, 1)(x) # Invert shadowgraph image (white vs black)
    x = tf.keras.layers.RandomRotation(1, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomZoom(32/128, fill_value=0.0, fill_mode='constant')(x)
    x = tf.keras.layers.RandomTranslation(0.3, 0.3, fill_mode='constant', fill_value=0.0)(x)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0))(x)
    x = tf.keras.layers.RandomContrast(0.5)(x)
    return x

def DenseNet(input_shape, num_classes):

    ## Init and Augmentation
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = augmentation_block(inputs)
    
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(128, (7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    ## DenseNet201
    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=48, growth_rate=32)
    x = transition_block(x, compression=0.5)
    x = dense_block(x, num_layers=32, growth_rate=32)

    # Final layers
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model

def load_model(config, num_classes):
    if int(config['training']['start']) > 0:
        return(tf.keras.models.load_model(config['training']['model_path'] + '/' + config['training']['model_name'] + '.keras'))
    return(init_model(num_classes, int(config['training']['image_size']), int(config['training']['image_size'])))


def init_model(num_classes, img_height, img_width):
    model = DenseNet([img_height, img_width, 1], num_classes)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    model.summary()
    return(model)


def train_model(model, config, train_ds, val_ds):
    csv_logger = tf.keras.callbacks.CSVLogger(config['training']['model_path'] + '/' + config['training']['model_name'] + '.log', append=False, separator=',')

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=int(config['training']['stop'])-int(config['training']['start']),
                        initial_epoch=int(config['training']['start']),
                        batch_size = int(config['training']['batchsize']),
                        callbacks=[csv_logger])
    
    return(model, history)


def init_ts(config):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        config['training']['scnn_dir'],
        interpolation='area',
        validation_split = config['training']['validationSetRatio'],
        subset = "both",
        seed = int(config['training']['seed']),
        image_size = (int(config['training']['image_size']), int(config['training']['image_size'])),
        batch_size = int(config['training']['batchsize']),
        color_mode = 'grayscale')
    
    return(train_ds, val_ds)


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = setup_logger('Training (main)', config)

    v_string = "V2024.05.22"
    session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
    
    logger.info(f"Starting CNN Model Training Script {v_string}")
    logger.debug(f"Session ID is {session_id}.")
    timer = {'init' : time()}

    # ## Load training and validation data sets
    train_ds, val_ds = init_ts(config)
    y = np.concatenate([y for x, y in val_ds], axis = 0)
    logger.info('Loaded training and validation datasets.')
    logger.debug(f"Datasets have {len(train_ds.class_names)} categories.")

    logger.debug('Attempting to load or initialize model.')
    timer['model_load_start'] = time()
    model = load_model(config, len(train_ds.class_names))
    timer['model_load_end'] = time()
    logger.debug('Model loaded successfully.')

    logger.info('Starting to train model.')
    
    timer['model_train_start'] = time()
    model, history = train_model(model, config, train_ds, val_ds)
    timer['model_train_end'] = time()
    logger.info('Model trained. Running post-processing steps.')

    model_save_pathname = config['training']['model_path'] + '/' + config['training']['model_name'] + '.keras'
    json_save_pathname = config['training']['model_path'] + '/' + config['training']['model_name'] + '.json'
    if os.path.exists(model_save_pathname):
        if config['training']['overwrite']:
            logger.info(f"Saved keras file exists for {config['training']['model_name']}. Overwrite is enabled so existing file is being removed.")
            delete_file(model_save_pathname, logger)
            logger.debug(f"Deleted file {model_save_pathname}.")
            if os.path.exists(json_save_pathname):
                delete_file(json_save_pathname, logger)
                logger.debug(f"Deleted file {json_save_pathname}.")
        else :
            config['training']['model_name'] = config['training']['model_name'] + '-' + session_id ## Append session ID to model_name from here on out!
            logger.warn(f"Saved keras file exists. Overwrite is not indicated so current model will be saved as {config['training']['model_name'] + '.keras'}.")
            model_save_pathname = config['training']['model_path'] + '/' + config['training']['model_name'] + '.keras'
            config['training']['model_path'] + '/' + config['training']['model_name'] + '.json'
    
    logger.debug(f"Saving model keras file to {config['training']['model_path']}.")
    timer['model_save_start'] = time()
    model.save(model_save_pathname)
    timer['model_save_end'] = time()
    logger.debug(f"Model saved to {config['training']['model_name'] + '.keras'}.")

    logger.debug('Running preditions on validation dataset.')
    predictions = model.predict(val_ds)
    prediction_matrix = pd.DataFrame(predictions, index = y, columns = train_ds.class_names)
    prediction_matrix.to_csv(config['training']['model_path'] + '/' + config['training']['model_name'] + ' predictions.csv')

    predictions = np.argmax(predictions, axis = -1)
    logger.debug('Developing confusion matrix from validation dataset.')
    confusion_matrix = tf.math.confusion_matrix(y, predictions)
    confusion_matrix = pd.DataFrame(confusion_matrix, index = train_ds.class_names, columns = train_ds.class_names)
    confusion_matrix.to_csv(config['training']['model_path'] + '/' + config['training']['model_name'] + ' confusion.csv')
    logger.debug(f"Confusion matrix saved to {config['training']['model_path'] + '/' + config['training']['model_name'] + ' confusion.csv'}")
    
    timer['close'] = time()
    
    logger.debug('Generating model saidecar.')
    ## Generate sidecar dictionary:
    sidecar = {
        'model_name' : config['training']['model_name'],
        'model_type' : config['training']['model_type'],
        'labels' : train_ds.class_names,
        'script_version' : v_string,
        'config' : config,
        'system_info' : {
            'System' : platform.system(),
            'Node' : platform.node(),
            'Release' : platform.release(),
            'Version' : platform.version(),
            'Machine' : platform.machine(),
            'Processor' : platform.processor()
        },
        'timings' : timer
    }
    
    logger.debug(f"Writing model sidecar to {config['training']['model_path']}.")
    json_object = json.dumps(sidecar, indent=4)
    with open(json_save_pathname, "w") as outfile:
        outfile.write(json_object)
        logger.debug("Sidecar writting finished.")
    
    logger.debug('Training finished.')
    sys.exit(0) # Successful close
