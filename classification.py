#!/usr/bin/env python3
"""Classification script for DPI-Scripts

Usage:
    ./classification.py <dir>

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
import sys
import shutil
import logging
import logging.config
import datetime
from time import time
from multiprocessing import Pool, Queue
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pandas as pd
import json
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Process


def is_file_above_minimum_size(file_path, min_size, logger):
    """
    Check if the file at file_path is larger than min_size bytes.

    :param file_path: Path to the file
    :param min_size: Minimum size in bytes
    :return: True if file size is above min_size, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    try:
        file_size = os.path.getsize(file_path)
        return file_size > min_size
    except OSError as e:
        logger.error(f"Error: {e}")
        return False


def delete_file(file_path, logger):
    """
    Delete the file at file_path.

    :param file_path: Path to the file to be deleted
    """
    
    try:
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            logger.debug(f"The folder '{file_path}' has been deleted.")
        else:
            os.remove(file_path)
            logger.debug(f"The file '{file_path}' has been deleted.")
    except FileNotFoundError:
        logger.debug(f"The file '{file_path}' does not exist.")
    except PermissionError:
        logger.warn(f"Permission denied: unable to delete '{file_path}'.")
    except OSError as e:
        logger.error(f"Error: {e}")


def setup_logger(name, config):
  # The name should be unique, so you can get in in other places
  # by calling `logger = logging.getLogger('com.dvnguyen.logger.example')
  logger = logging.getLogger(name) 
  logger.setLevel(logging.DEBUG) # the level should be the lowest level set in handlers

  log_format = logging.Formatter('[%(levelname)s] (%(process)d) %(asctime)s - %(message)s')
  if not os.path.exists(config['general']['log_path']):
    os.makedirs(config['general']['log_path'])
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(log_format)
  stream_handler.setLevel(logging.INFO)
  logger.addHandler(stream_handler)

  debug_handler = TimedRotatingFileHandler(f"{config['general']['log_path']}{name} debug.log", interval = 1, backupCount = 14)
  debug_handler.setFormatter(log_format)
  debug_handler.setLevel(logging.DEBUG)
  logger.addHandler(debug_handler)

  info_handler = TimedRotatingFileHandler(f"{config['general']['log_path']}{name} info.log", interval = 1, backupCount = 14)
  info_handler.setFormatter(log_format)
  info_handler.setLevel(logging.INFO)
  logger.addHandler(info_handler)

  error_handler = TimedRotatingFileHandler(f"{config['general']['log_path']}{name} error.log", interval = 1, backupCount = 14)
  error_handler.setFormatter(log_format)
  error_handler.setLevel(logging.ERROR)
  logger.addHandler(error_handler)
  return logger

if __name__ == "__main__":
    
    with open('config.json', 'r') as f:
        config = json.load(f)

    v_string = "V2024.05.22"
    session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
    logger = setup_logger('Classification (main)', config)
    logger.info(f"Starting Plankline Classification Script {v_string}")

    directory = sys.argv[1]
    if not os.path.exists(directory):
        logger.error(f'Specified path ({directory}) does not exist. Stopping.')
        sys.exit(1)


    if config['classification']['cpuonly']:
        logger.info("Disabling GPU support for this Tensorflow session.")
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
    

    # Load model
    model_path = f"../model/{config['classification']['model_name']}.keras"
    label_path = f"../model/{config['classification']['model_name']}.json"
    logger.info(f"Loading model from {model_path}.")
    logger.info(f"Loading model sidecar from {label_path}.")
    model = tf.keras.models.load_model(model_path)
    
    with open(label_path, 'r') as file:
        sidecar = json.load(file)
        logger.debug('Sidecar Loaded.')

    logger.info(f"Loaded keras model {config['classification']['model_name']} and sidecar JSON file.")
    
    # ### Setup Folders and run classification on each segment output
    segmentation_dir = os.path.abspath(directory)  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
    classification_dir = segmentation_dir.replace('segmentation', 'classification')  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
    classification_dir = classification_dir + '-' + config["classification"]["model_name"] # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg-Plankton
    
    logger.debug(f"Segmentation directory is {segmentation_dir}.")
    logger.debug(f"Classification directory is {classification_dir}.")
    
    os.makedirs(classification_dir, int(config['general']['dir_permissions']), exist_ok = True)
    
    root = [z for z in os.listdir(segmentation_dir) if z.endswith('zip')]
    
    logger.info(f"Found {len(root)} archives for potential processing.")
    for idx, r in enumerate(root):
        logger.debug(f"Archive {idx}: {r}")

    n = 1
    init_time = time()
    for r in root:
        start_time = time()
        if not is_file_above_minimum_size(segmentation_dir + '/' + r, 128, logger):
            logger.warn(f"File {r} either does not exist or does not meet minimum size requirements (128 B). Skipping to next file.")
            continue

        ##Unpack zip file
        r2 = r.replace(".zip", "")

        classification_output_filepath = classification_dir + '/' + r2 + '_' + 'prediction.csv'
        if os.path.exists(classification_output_filepath):
            if config['classification']['overwrite']:
                logger.info(f"classification file exists for {r2} and overwrite is enabled. Overwritting prior classification file.")
                delete_file(classification_output_filepath, logger)
            else:
                logger.info(f"classification file exists for {r2}, overwrite is contraindicated in the config file. Skipping.")

        logger.debug(f"Unpacking archive at {segmentation_dir + '/' + r} to destination {segmentation_dir + '/' + r2}.")
        shutil.unpack_archive(segmentation_dir + '/' + r, segmentation_dir + '/' + r2 + "/", 'zip')

        ## Load and preprocess images
        images = []
        image_files = []
        for img in os.listdir(segmentation_dir + '/' + r2):
            if img.endswith(('png', 'jpeg', 'jpg', 'tif', 'tiff')): 
                logger.debug(f"Loading image {img}.")
                image_files.append(img)
                img = tf.keras.preprocessing.image.load_img(segmentation_dir + '/' + r2 + '/' + img,
                                                            target_size=(int(config['classification']['image_size']), int(config['classification']['image_size'])),
                                                            color_mode='grayscale')
                img = np.expand_dims(img, axis=0)
                images.append(img)
            else :
                logger.debug(f"Skipping file {img}.")
        images = np.vstack(images)
        
        logger.debug("Finished loading images. Starting prediction.")
        predictions = model.predict(images, verbose = 0)
        prediction_labels = np.argmax(predictions, axis=-1)
        prediction_labels = [sidecar['labels'][i] for i in prediction_labels]
        df = pd.DataFrame(predictions, index=image_files)
        #df_short = pd.DataFrame(prediction_labels, index=image_files)
        
        
        df.columns = sidecar['labels']
        df.to_csv(classification_output_filepath, index=True, header=True, sep=',')
        #df_short.columns = ['prediction']
        #df_short.to_csv(classification_dir + '/' + r2 + '_' + 'predictionlist.csv', index=True, header=True, sep=',')
        
        logger.debug('Cleaning up unpacked archive files.')
        delete_file(segmentation_dir + '/' + r2 + "/", logger)
        end_time = time()
        logger.info(f"Processed {n} of {len(root)} files.\t\t Iteration: {end_time-start_time:.2f} seconds\t Estimated remainder: {(end_time - init_time)/n*(len(root)-n) / 60:.1f} minutes.\t Elapsed time: {(end_time - init_time)/60:.1f} minutes.")
        n+=1
    logger.info(f"Finished classification. Total time: {(end_time - init_time)/60:.1f} minutes.")
    sys.exit(0) # Successful close
