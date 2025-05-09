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
import platform
from .functions import *


def loadModel(config, logger):
    """
    Helper function to load model and sidecar. 
    """
    model_path = config['classification']['model_dir'] + os.path.sep + config['classification']['model_name'] + ".keras"
    label_path = config['classification']['model_dir'] + os.path.sep + config['classification']['model_name'] + ".json"
    logger.info(f"Loading model from {model_path}.")
    logger.info(f"Loading model sidecar from {label_path}.")
    model = tf.keras.models.load_model(model_path)

    if config['classification']['feature_space']:
        ## Modify model for feature extraction:
        # Remove final softmax activation to expose penultimate dense layer. Generate as new model object.
        x = model.layers[-2].output 
        model = tf.keras.models.Model(inputs = model.input, outputs = x)
    
    with open(label_path, 'r') as file:
        sidecar = json.load(file)
        logger.debug('Sidecar Loaded.')

    logger.info(f"Loaded keras model {config['classification']['model_name']} and sidecar JSON file.")

    return model, sidecar


def mainClassifcation(config, logger):
    """
    Main function for classification.
    """
    v_string = "V2024.05.22"
    #session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
    logger.info(f"Starting Plankline Classification Script {v_string}")
    timer = {'init' : time()}

    if not os.path.exists(config['segmentation_dir']):
        logger.error(f'Specified path ({config["segmentation_dir"]}) does not exist. Stopping.')
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
            logger.info("Unable to disable GPU device. Exception thrown, continuing.")
            pass
    
    # Load model
    timer['model_load_start'] = time()
    model, sidecar = loadModel(config, logger)
    timer['model_load_end'] = time()

    # ### Setup Folders and run classification on each segment output
    timer['folder_prepare_start'] = time()

    logger.debug(f"Segmentation directory is {config['segmentation_dir']}.")
    logger.debug(f"Classification directory is {config['classification_dir']}.")
    
    try:
        os.makedirs(config['classification_dir'], int(config['general']['dir_permissions']), exist_ok = True)
    except PermissionError:
        logger.error(f"Permission denied: Unable to create classification directory '{config['classification_dir']}'.")
    except OSError as e:
        logger.error(f"Error creating directory '{config['classification_dir']}': {e}")
    
    if config['classification']['from_database']:
        
        # TODO Define root or otherwise make this section work!
        sys.exit(1)

        logger.info(f"Found {len(root)} archives for potential processing.")
        timer['folder_prepare_end'] = time()

        init_time = time()
        timer['processing_start'] = time()
        start_time = time()
        runClassifier(r, model, sidecar, config, logger)
        end_time = time()
            
        logger.info(f"Finished classification. Total time: {(end_time - init_time)/60:.1f} minutes.")
        timer['processing_end'] = time()
    else:
        root = [z for z in os.listdir(config['segmentation_dir']) if z.endswith('.zip')]
        
        logger.info(f"Found {len(root)} archives for potential processing.")
        for idx, r in enumerate(root):
            logger.debug(f"Archive {idx}: {r}")
        timer['folder_prepare_end'] = time()

        n = 1
        init_time = time()
        timer['processing_start'] = time()
        for r in root:
            start_time = time()
            runClassifier(r, model, sidecar, config, logger)
            logger.debug('Cleaning up unpacked archive files.')
            delete_file(config['segmentation_dir'] + os.path.sep + r.replace(".zip", "") + os.path.sep, logger)
            end_time = time()
            logger.info(f"Processed {n} of {len(root)} files.\t\t Iteration: {end_time-start_time:.2f} seconds\t Estimated remainder: {(end_time - init_time)/n*(len(root)-n) / 60:.1f} minutes.\t Elapsed time: {(end_time - init_time)/60:.1f} minutes.")
            n+=1
            
        logger.info(f"Finished classification. Total time: {(end_time - init_time)/60:.1f} minutes.")
        timer['processing_end'] = time()

    classificationSidecar = {
        'directory' : config['segmentation_dir'],
        'nFiles' : len(root),
        'script_version' : v_string,
        'sessionid': str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', ''),
        'config' : config,
        'labels' : sidecar['labels'],
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

    return classificationSidecar


def runClassifier(r, model, sidecar, config, logger):
    if not is_file_above_minimum_size(config['segmentation_dir'] + os.path.sep + r, 128, logger):
        logger.warn(f"File {r} either does not exist or does not meet minimum size requirements (128 B). Skipping to next file.")
        return(1)

    ##Unpack zip file
    r2 = r.replace(".zip", "")

    classification_output_filepath = config['classification_dir'] + os.path.sep + r2 + ' prediction.csv'
    if os.path.exists(classification_output_filepath):
        if config['classification']['overwrite']:
            logger.info(f"classification file exists for {r2} and overwrite is enabled. Overwritting prior classification file.")
            delete_file(classification_output_filepath, logger)
        else:
            logger.info(f"classification file exists for {r2}, overwrite is contraindicated in the config file. Skipping.")

    logger.debug(f"Unpacking archive at {config['segmentation_dir'] + os.path.sep + r} to destination {config['segmentation_dir'] + os.path.sep + r2}.")
    shutil.unpack_archive(config['segmentation_dir'] + os.path.sep + r, config['segmentation_dir'] + os.path.sep + r2 + os.path.sep, 'zip')

    ## Load and preprocess images
    images = []
    image_files = []
    valid_extensions = tuple(config['classification']['image_extensions'])
    logger.debug(f"Valid extensions: {', '.join(valid_extensions)}")

    for img in os.listdir(config['segmentation_dir'] + os.path.sep + r2):
        if img.endswith(valid_extensions): 
            logger.debug(f"Loading image {img}.")
            image_files.append(img)
            img = tf.keras.preprocessing.image.load_img(config['segmentation_dir'] + os.path.sep + r2 + os.path.sep + img,
                        target_size = (int(config['classification']['image_size']), int(config['classification']['image_size'])),
                        color_mode = 'grayscale')
            img = np.expand_dims(img, axis = 0)
            images.append(img)
        else :
            logger.debug(f"Skipping file {img}.")
    images = np.vstack(images)
        
    logger.debug("Finished loading images. Starting prediction.")
    predictions = model.predict(images, verbose = 0)

    df = pd.DataFrame(predictions, index = image_files) 
    df.columns = sidecar['labels']
    df.to_csv(classification_output_filepath, index = True, header = True, sep = ',')