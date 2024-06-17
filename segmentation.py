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

import sys
import configparser
from PIL import Image
import cv2
import os
import numpy as np
import csv
from multiprocessing import Process, Queue
import tqdm
import shutil
import random
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from time import time

class Frame:
    def __init__(self, fpath, name, frame, n):
        self.fpath = fpath  # default is 0 for primary camera
        self.name = name
        self.frame = frame
        self.n = n

    def read(self):
        return self.frame

    def get_n(self):
        return self.n

    def get_name(self):
        return self.name

    def update(self, newframe):
        self.frame = newframe



def process_frame(q, config): ## TODO: write metadata file
    """
    This function processes each frame (provided as cv2 image frame) for flatfielding and segmentation. The steps include
    1. Flatfield intensities as indicated
    2. Segment the image using cv2 MSER algorithmn.
    3. Remove strongly overlapping bounding boxes
    4. Save cropped targets.
    """
    logger = setup_logger('worker', config)
    logger.debug('Started worker thread.')

    while True:
        frame = q.get()
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

        name = frame.get_name()
        n = frame.get_n()
        stats = []

        if config['segmentation']['diagnostic']:
            logger.debug('Saving diagnostic images.')
            cv2.imwrite(f'{name}{n:06}-qualtilefield.jpg', gray)
            cv2.imwrite(f'{name}{n:06}-threshold.jpg', thresh)

        with open(f'{name[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
            logger.debug(f"Writing to statistics.csv. Found {len(cnts)} ROIs.")
            outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
            for i in range(len(cnts)):
                x,y,w,h = cv2.boundingRect(cnts[i])
                if 2*w + 2*h > int(config['segmentation']['min_perimeter_statsonly']):
                    if len(cnts[i]) >= 5:  # Minimum number of points required to fit an ellipse
                        ellipse = cv2.fitEllipse(cnts[i])
                        center, axes, angle = ellipse
                        major_axis_length = max(axes)
                        minor_axis_length = min(axes)
                    else :
                        major_axis_length = -1
                        minor_axis_length = -1
                    #mean_gray_value = np.mean(gray[y:(y+h), x:(x+w)])

                    if 2*w + 2*h >= int(config['segmentation']['min_perimeter']) and 2*w + 2*h <= int(config['segmentation']['max_perimeter']) :
                        
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
                        im_padded.save(f"{name}{n:06}-{i:06}.png")
                    stats = [n, i, x + w/2, y + h/2, w, h, major_axis_length, minor_axis_length]
                    outwritter.writerow(stats)
                

def process_avi(avi_path, segmentation_dir, config, q):
    """
    This function will take an avi filepath as input and perform the following steps:
    1. Create output file structures/directories
    2. Load each frame, pass it through flatfielding and sequentially save segmented targets
    """

    # segmentation_dir: /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    _, filename = os.path.split(avi_path)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    statistics_filepath = output_path[:-1] + ' statistics.csv'
    
    if is_file_above_minimum_size(statistics_filepath, 0):
        if config['segmentation']['overwrite']:
            logger.info(f"Config file enables overwriting, removing files for {filename}.")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            if os.path.exists(output_path[:-1] + '.zip'):
                os.remove(output_path[:-1] + '.zip')
            if os.path.exists(output_path[:-1] + ' statistics.csv'):
                os.remove(output_path[:-1] + ' statistics.csv')

        else:
            logger.info(f"Overwritting is not allowed and prior statistics file exists. Skipping {filename}.")
            return

    try:
        os.makedirs(output_path, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied when making directory {output_path}.")

    video = cv2.VideoCapture(avi_path)
    if not video.isOpened():
        return
    
    with open(statistics_filepath, 'a', newline='\n') as outcsv:
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        outwritter.writerow(['frame', 'crop', 'x', 'y', 'w', 'h', 'major_axis', 'minor_axis'])

    n = 1 # Frame count
    while True:
        ret, frame = video.read()
        if ret:
            if not frame is None:
                q.put(Frame(avi_path, output_path, frame, n), block = True)
                n += 1 # Increment frame counter.
        else:
            break

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

    debug_handler = TimedRotatingFileHandler(f"{config['general']['log_path']}segmentation {name} debug.log", interval = 1, backupCount = 14)
    debug_handler.setFormatter(log_format)
    debug_handler.setLevel(logging.DEBUG)
    logger.addHandler(debug_handler)

    info_handler = TimedRotatingFileHandler(f"{config['general']['log_path']}segmentation {name} info.log", interval = 1, backupCount = 14)
    info_handler.setFormatter(log_format)
    info_handler.setLevel(logging.INFO)
    logger.addHandler(info_handler)

    error_handler = TimedRotatingFileHandler(f"{config['general']['log_path']}segmentation {name} error.log", interval = 1, backupCount = 14)
    error_handler.setFormatter(log_format)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    return logger


def is_file_above_minimum_size(file_path, min_size):
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
        print(f"Error: {e}")
        return False


if __name__ == "__main__":

    directory = sys.argv[1]
    if not os.path.exists(directory):
        stop(f'Specified path ({directory}) does not exist. Stopping.')

    with open('config.json', 'r') as f:
        config = json.load(f)

    v_string = "V2024.05.21"

    logger = setup_logger('main', config)
    logger.info(f"Starting segmentation script {v_string}")

    ## Determine directories
    raw_dir = os.path.abspath(directory) # /media/plankline/Data/raw/Camera0/test1
    segmentation_dir = raw_dir.replace("raw", "analysis") # /media/plankline/Data/analysis/Camera1/Transect1
    segmentation_dir = segmentation_dir.replace("camera0/", "camera0/segmentation/") # /media/plankline/Data/analysis/Camera1/Transect1
    segmentation_dir = segmentation_dir.replace("camera1/", "camera1/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    segmentation_dir = segmentation_dir.replace("camera2/", "camera2/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
    segmentation_dir = segmentation_dir.replace("camera3/", "camera3/segmentation/") # /media/plankline/Data/analysis/Camera1/segmentation/Transect1
        
    segmentation_dir = segmentation_dir + f"-{config['segmentation']['basename']}" # /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    logger.debug(f"Segmentation directory: {segmentation_dir}")
    os.makedirs(segmentation_dir, int(config['general']['dir_permissions']), exist_ok = True)

    ## Find files to process:
    # AVI videos
    avis = []
    avis = [os.path.join(raw_dir, avi) for avi in os.listdir(raw_dir) if avi.endswith(".avi")]
    logger.info(f"Number of AVIs found: {len(avis)}")

    for idx, av in enumerate(avis):
        logger.debug(f"Found AVI file {idx}: {av}.")

    ## Prepare workers for receiving frames
    num_threads = os.cpu_count() - 1
    max_queue = num_threads * 8 # Prepare 4 frames per thread. TODO: test memory vs performance considerations. UPDATE: 4 still seems okay on my laptop.
    q = Queue(maxsize=int(max_queue))
    workers = []

    for i in range(num_threads):
        worker = Process(target=process_frame, args=(q, config,), daemon=True)
        workers.append(worker)
        worker.start()
        
    logger.debug(f"Starting {num_threads} processing threads.")
    logger.debug(f"Initialized queue with size = {max_queue}.")

    if len(avis) > 0:
        logger.info(f'Starting processing on {len(avis)} AVI files.')
        n = 1
        init_time = time()
        for av in avis:
            if is_file_above_minimum_size(av, 1024):
                start_time = time()
                process_avi(av, segmentation_dir, config, q)
                end_time = time()
                logger.info(f"Processed {n} of {len(avis)} files.\t\t Iteration: {end_time-start_time:.2f} seconds\t Estimated remainder: {(end_time - init_time)/n*(len(avis)-n) / 60:.1f} minutes.\t Elapsed time: {(end_time - init_time)/60:.1f} minutes.")
                n+=1
            else:
                logger.warn(f"File {av} either does not exist or does not meet minimum size requirements (1 kB). Skipping to next file.")

    logger.info('Joining worker processes.')
    for worker in workers:
        worker.join(timeout=1)
    
    if len(avis) > 0:
        logger.info('Archiving results and cleaning up.')
        for av in avis:
            _, filename = os.path.split(av)
            output_path = segmentation_dir + os.path.sep + filename + os.path.sep
            logger.debug(f"Compressing to archive {filename + '.zip.'}")
            if is_file_above_minimum_size(segmentation_dir + os.path.sep + filename + '.zip', 0):
                if config['segmentation']['overwrite']:
                    logger.warn(f"archive exists for {filename} and overwritting is allowed. Deleting old archive.")
                    os.remove(segmentation_dir + os.path.sep + filename + '.zip')
                else:
                    logger.warn(f"archive exists for {filename} and overwritting is allowed. Skipping Archiving")
                    continue

            shutil.make_archive(segmentation_dir + os.path.sep + filename, 'zip', output_path)
            if not config['segmentation']['diagnostic']:
                logger.debug(f"Cleaning up output path: {output_path}.")
                shutil.rmtree(output_path, ignore_errors=True)

    logger.info(f"Finished segmentation. Total time: {(time() - init_time)/60:.1f} minutes.")
    sys.exit(0) # Successful close

