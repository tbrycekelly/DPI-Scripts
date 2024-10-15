#!/usr/bin/env python3
"""Segmentation script for DPI-Scripts

Usage:
    ./segmentation.py <dir>

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
from time import time
import numpy as np
import csv
from PIL import Image
import os
import json
from logging.handlers import TimedRotatingFileHandler
import cv2
import concurrent.futures


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
    """
    A setup function to build a logger of a given name (should be unique up to use).
    """
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


class Frame:
    def __init__(self, fpath, name, frame, n, filename):
        self.fpath = fpath
        self.name = name
        self.frame = frame
        self.n = n
        self.filename = filename

    def read(self):
        return self.frame

    def get_n(self):
        return self.n

    def get_name(self):
        return self.name

    def update(self, newframe):
        self.frame = newframe
    
    def get_filename(self):
        return self.filename


def process_frame(frame, config, logger):
    """
    Function for processing a single frame object:
        Flatfielding
        Thresholding
        Contouring
        Segmentation
        Statistics
        ROIs
    """
    logger.debug('Started worker thread.')

    logger.debug(f"Pulled frame from queue. Processing {frame.get_name()}.")
        
    ## Read img and flatfield
    gray = cv2.cvtColor(frame.read(), cv2.COLOR_BGR2GRAY)
    gray = np.array(gray)
        
    field = np.quantile(gray, q = float(config['segmentation']['flatfield_q']), axis = 0)
    gray = (gray / field.T * 255.0)
    gray = gray.clip(0,255).astype(np.uint8)

    # Apply Otsu's threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
    path = frame.get_name()
    n = frame.get_n()
    filename = frame.get_filename()
    stats = []
    grayAnnotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Open statistics file and iterate through all identified ROIs.
    with open(f'{path[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
        logger.debug(f"Writing to statistics.csv. Found {len(cnts)} ROIs.")
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        for i in range(len(cnts)):
            x,y,w,h = cv2.boundingRect(cnts[i])
            area = cv2.contourArea(cnts[i])

            # If ROI is of useful minimum size.
            if 2*w + 2*h >= int(config['segmentation']['min_perimeter_statsonly']):
                if len(cnts[i]) >= 5:  # Minimum number of points required to fit an ellipse
                    ellipse = cv2.fitEllipse(cnts[i])
                    center, axes, angle = ellipse
                    major_axis_length = round(max(axes),1)
                    minor_axis_length = round(min(axes),1)
                else :
                    major_axis_length = -1
                    minor_axis_length = -1

                # If ROI is within size limits for saving an image. 
                if 2*w + 2*h >= int(config['segmentation']['min_perimeter']) and 2*w + 2*h <= int(config['segmentation']['max_perimeter']):
                    
                    if config['segmentation']['diagnostic']:
                        cv2.rectangle(grayAnnotated, (x, y), (x+w, y+h), (0,0,255), 1)

                    # Save crop as a square ROI. Need to determine size and padding.
                    size = max(w, h)
                    im = Image.fromarray(gray[y:(y+h), x:(x+w)])
                    im_padded = Image.new(im.mode, (size, size), (255))
                        
                    if (w > h):
                        left = 0
                        top = (size - h)//2
                    else:
                        left = (size - w)//2
                        top = 0
                    im_padded.paste(im, (left, top))
                    im_padded.save(f"{path}{filename}-{n:06}-{i:06}.png")
                
                # Write stats to file:
                stats = [n, i, x + w/2, y + h/2, w, h, major_axis_length, minor_axis_length, area]
                outwritter.writerow(stats)

    # Save optional diagnsotic images before returning.
    if config['segmentation']['diagnostic']:
        logger.debug('Saving diagnostic images.')
        cv2.imwrite(f'{path}{filename}-{n:06}-qualtilefield.jpg', gray)
        cv2.imwrite(f'{path}{filename}-{n:06}-annotated.jpg', grayAnnotated)
        cv2.imwrite(f'{path}{filename}-{n:06}-threshold.jpg', thresh)

    return True
                

def process_avi(segmentation_dir, config, avi_path):
    """
    This function will take an avi filepath as input and perform the following steps:
    1. Create output file structures/directories
    2. Load each frame, pass it through flatfielding and sequentially save segmented targets
    """
    # segmentation_dir: /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    logger = setup_logger('Segmentation (Worker)', config)
    _, filename = os.path.split(avi_path)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    statistics_filepath = output_path[:-1] + ' statistics.csv'
    logger = logging.getLogger('Segmentation (Worker)')
    
    if is_file_above_minimum_size(statistics_filepath, 0, logger):
        if config['segmentation']['overwrite']:
            logger.info(f"Config file enables overwriting, removing files for {filename}.")
            if os.path.exists(output_path):
                delete_file(output_path, logger)
            if os.path.exists(output_path[:-1] + '.zip'):
                delete_file(output_path[:-1] + '.zip', logger)
            if os.path.exists(output_path[:-1] + ' statistics.csv'):
                delete_file(output_path[:-1] + ' statistics.csv', logger)

        else:
            logger.info(f"Overwritting is not allowed and prior statistics file exists. Skipping {filename}.")
            return

    try:
        os.makedirs(output_path, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied when making directory {output_path}.")

    # Open video file, initialize statistcs file, and start going through frames.
    video = cv2.VideoCapture(avi_path)
    if not video.isOpened():
        logger.error(f"Issue openning video {avi_path}.")
        return
        
    with open(statistics_filepath, 'a', newline='\n') as outcsv:
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        outwritter.writerow(['frame', 'crop', 'x', 'y', 'w', 'h', 'major_axis', 'minor_axis', 'area'])

    n = 1 # Frame count
    while True:
        ret, frame = video.read()
        if ret:
            if not frame is None:
                process_frame(Frame(avi_path, output_path, frame, n, filename.replace('.avi', '')), config, logger)
                n += 1 # Increment frame counter.
        else:
            break


def cleanupFile(av, segmentation_dir, config):
    """
    cleanupFile: Standalone function to move a segmentated directory of ROIs into a compressed
    zip archive. It then deletes the folder if not in diagnostic mode. This function allows for
    multithreaded operation.
    """
    logger = setup_logger('Segmentation (Worker)', config)
    _, filename = os.path.split(av)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    logger.debug(f"Compressing to archive {filename + '.zip.'}")
    if is_file_above_minimum_size(segmentation_dir + os.path.sep + filename + '.zip', 0, logger):
        if config['segmentation']['overwrite']:
            logger.warn(f"archive exists for {filename} and overwritting is allowed. Deleting old archive.")
            delete_file(segmentation_dir + os.path.sep + filename + '.zip', logger)
        else:
            logger.warn(f"archive exists for {filename} and overwritting is allowed. Skipping Archiving")

    shutil.make_archive(segmentation_dir + os.path.sep + filename, 'zip', output_path)
    if not config['segmentation']['diagnostic']:
        logger.debug(f"Cleaning up output path: {output_path}.")
        delete_file(output_path, logger)



def constructSegmentationDir(rawPath, config):
    """
    Helper function to standardize the directory construction for the segmentation output given a configuration
    and raw file path.
    """
    segmentationDir = rawPath.replace("raw", "analysis") # /media/plankline/Data/analysis/Camera1/Transect1
    segmentationDir = segmentationDir.replace("camera0/", "camera0/segmentation/") # /media/plankline/Data/analysis/Camera1/Transect1
    segmentationDir = segmentationDir.replace("camera1/", "camera1/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    segmentationDir = segmentationDir.replace("camera2/", "camera2/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    segmentationDir = segmentationDir.replace("camera3/", "camera3/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
        
    segmentationDir = segmentationDir + f"-{config['segmentation']['basename']}" # /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    logger.debug(f"Segmentation directory: {segmentationDir}")
    try:
        os.makedirs(segmentationDir, int(config['general']['dir_permissions']), exist_ok = True)
    except PermissionError:
        logger.error(f"Permission denied: Unable to create segmentation directory '{directory_path}'.")
        sys.exit(1)
    except OSError as e:
        # Catch any other OS-related errors
        logger.error(f"Error creating directory '{directory_path}': {e}")
        sys.exit(1)

    return(segmentationDir)


def findVideos(raw_dir, config, logger):
    """
    Helper function to search for availabile video files in a directory.
    """
    avis = []
    valid_extensions = tuple(config['segmentation']['video_extensions'])
    logger.debug(f"Valid extensions: {', '.join(valid_extensions)}")
    avis = [os.path.join(raw_dir, avi) for avi in os.listdir(raw_dir) if avi.endswith(valid_extensions)]
    logger.info(f"Number of videos found: {len(avis)}")

    for idx, av in enumerate(avis):
        logger.debug(f"Found video file {idx}: {av}.")

    return(avis)


def main(directory, config, logger):
    """
    The main access function for segmentation. Can be called from external scripts, but designed initially
     to be called from __main__.
    """
    v_string = "V2024.10.14"
    logger.info(f"Starting segmentation script {v_string}")

    if not os.path.exists(directory):
        logger.error(f'Specified path ({directory}) does not exist. Stopping.')
        sys.exit(1)

    ## Determine directories
    raw_dir = os.path.abspath(directory) # /media/plankline/Data/raw/Camera0/test1
    segmentation_dir = constructSegmentationDir(raw_dir, config)

    ## Find files to process:
    avis = findVideos(raw_dir, config, logger)
    
    ## Prepare workers for receiving frames
    num_threads = min(os.cpu_count() - 2, len(avis))
    logger.info(f"Starting processing with {num_threads} processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {executor.submit(process_avi, segmentation_dir, config, filename): filename for filename in avis}
    
        # Wait for all tasks to complete
        init_time = time()
        n = 1
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                future.result()  # Get the result of the computation
            except Exception as exc:
                logger.error(f'Processing {filename} generated an exception: {exc}')
            else:
                end_time = time()
                logger.info(f"Processed {n} of {len(avis)} files.\t\t Estimated remainder: {(end_time - init_time)/n*(len(avis)-n) / 60:.1f} minutes.\t Elapsed time: {(end_time - init_time)/60:.1f} minutes.")
                n += 1

    logger.info('Archiving results and cleaning up.') # Important to isolate processing and cleanup since the threads don't know when everything is done processing.
    with concurrent.futures.ProcessPoolExecutor(max_workers = num_threads) as executor:
        future_to_file = {executor.submit(cleanupFile, filename, segmentation_dir, config): filename for filename in avis}
    
        # Wait for all tasks to complete
        init_time2 = time()
        n = 1
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                future.result()  # Get the result of the computation
            except Exception as exc:
                logger.error(f'Processing {filename} generated an exception: {exc}')
            else:
                end_time = time()
                logger.info(f"Cleaning up {n} of {len(avis)} files.\t\t Estimated remainder: {(end_time - init_time2)/n*(len(avis)-n) / 60:.1f} minutes.\t Elapsed time: {(end_time - init_time2)/60:.1f} minutes.")
                n += 1

    logger.info(f"Finished segmentation. Total time: {(time() - init_time)/60:.1f} minutes.")
    sys.exit(0) # Successful close


if __name__ == "__main__":
    """
    Entrypoint for script when run from the command line.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    v_string = "V2024.10.14"
    logger = setup_logger('Segmentation (Main)', config)

    directory = sys.argv[1]
    main(directory, config, logger)