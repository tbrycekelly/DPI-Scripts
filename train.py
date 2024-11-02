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
import logging
import logging.config
from time import time
from multiprocessing import Pool
import datetime
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import json
import platform
import sys
from logging.handlers import TimedRotatingFileHandler

from functions import *
#from ConvNet import *
from ResNet import *


def load_model(config, num_classes):
    if int(config['training']['start']) > 0:
        return(tf.keras.models.load_model(config['training']['model_path'] + '/' + config['training']['model_name'] + '.keras'))
    return(init_model(num_classes, int(config['training']['image_size']), int(config['training']['image_size'])))


def init_model(num_classes, img_height, img_width):
    model = Model([img_height, img_width, 1], num_classes)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    model.summary()
    return(model)


def train_model(model, config, train_ds, val_ds, devices):
    csv_logger = tf.keras.callbacks.CSVLogger(
        config['training']['model_path'] + '/' + config['training']['model_name'] + '.log',
         append = False,
          separator = ','
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = config['training']['model_path'] + os.path.sep + config['training']['model_name'] + '_checkpoint.keras',
        save_best_only = True,
        monitor = 'val_loss',
        mode = 'min',
        save_weights_only = False,
        save_freq = 5
    )

    with tf.device(devices):
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=int(config['training']['stop'])-int(config['training']['start']),
                            initial_epoch=int(config['training']['start']),
                            batch_size = int(config['training']['batchsize']),
                            callbacks=[csv_logger, checkpoint_callback])
    
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


def getTensorflowDevices(logger):
    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        logger.info(f"Found {len(gpus)} Tensorflow-compatible GPUs.")
        for idx, g in enumerate(gpus):
            logger.debug(f"Device {idx}: {g}")
        return gpus
    else:
        logger.warn(f"No (compatible) GPUs found, defaulting to CPU execution (n={len(devices)}).")
        for idx, cpu in enumerate(devices):
            logger.debug(f"Device {idx}: {cpu}")
    return devices


def mainTrain(config, logger):

    v_string = "V2024.05.22"
    session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
    
    logger.info(f"Starting CNN Model Training Script {v_string}")
    logger.debug(f"Session ID is {session_id}.")
    timer = {'init' : time()}

    deviceList = getTensorflowDevices(logger)

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
    
    ## Train model
    timer['model_train_start'] = time()
    model, history = train_model(model, config, train_ds, val_ds, deviceList)
    timer['model_train_end'] = time()
    logger.info('Model trained. Running post-processing steps.')

    ## Post training steps
    model_save_pathname = config['training']['model_path'] + os.path.sep + config['training']['model_name'] + '.keras'
    json_save_pathname = config['training']['model_path'] + os.path.sep + config['training']['model_name'] + '.json'

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
            model_save_pathname = config['training']['model_path'] + os.path.sep + config['training']['model_name'] + '.keras'
            config['training']['model_path'] + os.path.sep + config['training']['model_name'] + '.json'
    
    logger.debug(f"Saving model keras file to {config['training']['model_path']}.")
    timer['model_save_start'] = time()
    model.save(model_save_pathname)
    timer['model_save_end'] = time()
    logger.debug(f"Model saved to {config['training']['model_name'] + '.keras'}.")
    
    logger.debug('Running preditions on validation dataset.')
    predictions = model.predict(val_ds)
    prediction_matrix = pd.DataFrame(predictions, index = y, columns = train_ds.class_names)
    prediction_matrix.to_csv(config['training']['model_path'] + os.path.sep + config['training']['model_name'] + ' predictions.csv')
    
    logger.debug('Developing confusion matrix from validation dataset.')
    predictions = np.argmax(predictions, axis = -1)
    confusion_matrix = tf.math.confusion_matrix(y, predictions)
    confusion_matrix = pd.DataFrame(confusion_matrix, index = train_ds.class_names, columns = train_ds.class_names)
    confusion_matrix.to_csv(config['training']['model_path'] + os.path.sep + config['training']['model_name'] + ' confusion.csv')
    logger.debug(f"Confusion matrix saved to {config['training']['model_path'] + os.path.sep + config['training']['model_name'] + ' confusion.csv'}")
    
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
    
    logger.debug(f"Writing model sidecar to {json_save_pathname}.")
    json_object = json.dumps(sidecar, indent=4)
    with open(json_save_pathname, "w") as outfile:
        outfile.write(json_object)
        logger.debug("Sidecar writting finished.")
    
    logger.debug('Training finished.')
    sys.exit(0) # Successful close


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = setup_logger('Training (main)', config)

    mainTrain(config, logger)

