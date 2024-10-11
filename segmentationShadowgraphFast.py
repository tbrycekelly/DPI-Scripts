#!/usr/bin/env python3
"""Segmentation script for DPI-Scripts (shadowgraph glider version)

Usage:
    ./segmentationShadowgraph.py <dir>

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
from PIL import Image
import cv2
import os
import numpy as np
import csv
from multiprocessing import Process, Queue
import shutil
import random
import logging
import json
from time import time
import shutil
import logging.config
from logging.handlers import TimedRotatingFileHandler
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


class Frame:
    def __init__(self, fpath, name, frame, cal, n):
        self.fpath = fpath  # default is 0 for primary camera
        self.name = name
        self.frame = frame
        self.n = n
        self.cal = cal

    def read(self):
        return self.frame

    def get_n(self):
        return self.n

    def get_name(self):
        return self.name

    def update(self, newframe):
        self.frame = newframe

    def calibration(self):
        return self.cal



def process_frame(frame, config):
    """
    This function processes each frame (provided as cv2 image frame) for flatfielding and segmentation. The steps include
    1. Flatfield intensities as indicated
    2. Segment the image using cv2 MSER algorithmn.
    3. Remove strongly overlapping bounding boxes
    4. Save cropped targets.
    """
    logger = setup_logger('Shadowgraph Segmentation (worker)', config)
    logger.debug("Process started.")
    
    logger.debug(f"Pulled frame from queue. Processing {frame.get_name()} {frame.get_n()}.")
        
    image = np.array(cv2.imread(frame.read()))
    # Check for compatible sizes:
    if image.shape != frame.calibration().shape:
        logger.debug(f"Image sizes for the frame and clibration images are not the same: {frame.get_name()} {frame.get_n()}.")
    else:
        gray = image / frame.calibration() * 255
        gray = gray.clip(0,255).astype(np.uint8)

        gray = ~gray
        mask = np.zeros(gray.shape[:2], dtype="uint8")
        cv2.circle(mask, (gray.shape[1]//2, gray.shape[0]//2), 1100, 255, -1)
        gray = cv2.bitwise_and(gray, gray, mask = mask) # Mask
        gray = ~gray
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            
        gray = np.array(gray)
        # Rescale to enahnce contrast:
        #gray = 16.0 * (gray // 16) # posterize to 16 values.
        gray = 255.0 * (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
        gray = gray.clip(0,255).astype(np.uint8)

            # Apply Otsu's threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        name = frame.get_name()
        n = frame.get_n()
        logger.debug(f"Thresholding frame {n}.")
        stats = []

        if config['segmentation']['diagnostic']:
            logger.debug(f"Diagnostic mode, saving threshold image and quantiledfiled image.")
            cv2.imwrite(f'{name}{n:06}-qualtilefield.jpg', gray)
            cv2.imwrite(f'{name}{n:06}-threshold.jpg', thresh)

        with open(f'{name[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
            outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
            for i in range(len(cnts)):
                x,y,w,h = cv2.boundingRect(cnts[i])
                area = cv2.contourArea(cnts[i])
                if 2*w + 2*h > int(config['segmentation']['min_perimeter_statsonly']):
                    if len(cnts[i]) >= 5:  # Minimum number of points required to fit an ellipse
                        ellipse = cv2.fitEllipse(cnts[i])
                        center, axes, angle = ellipse
                        major_axis_length = round(max(axes),1)
                        minor_axis_length = round(min(axes),1)
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
                    stats = [n, i, x + w/2, y + h/2, w, h, major_axis_length, minor_axis_length, area]
                    outwritter.writerow(stats)
        logger.debug(f"Done with frame {n}.")
                


def process_image_dir(img_path, segmentation_dir, config):
    """
    This function will take an image folder as input and perform the following steps:
    1. Create output file structures/directories
    2. Load each frame, pass it through flatfielding and sequentially save segmented targets
    """

    # segmentation_dir: /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    _, filename = os.path.split(img_path)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    os.makedirs(output_path, exist_ok=True)
    logger.debug(f"Created directory {output_path} if not already existing.")

    with open(f'{output_path[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
        logger.info(f"Initialized metrics file for {filename}.")
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        outwritter.writerow(['frame', 'crop', 'x', 'y', 'w', 'h'])

    logger.debug(f"Reading in calibration image {config['segmentation']['calibration_image']}.")
    bkg = np.array(cv2.imread(config['segmentation']['calibration_image']))
    
    for f in os.listdir(img_path):
      if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
          logger.debug(f"Processing image {f}.")
          process_frame(Frame(f, output_path, img_path + os.path.sep + f, bkg, f), config)
      else:
        logger.debug(f"Skipped reading non-image file {f}.") 


def generate_median_image(directory, output_dir):
    """
    
    """
    logger = logging.getLogger('Shadowgraph Segmentation (main)')
    # Get a list of all image file names in the directory
    image_files = [file for file in os.listdir(directory) if file.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    
    if not image_files:
        logger.error("No image files found in the directory. Cannot generate calibration image!")
        return
    nImages = min([100, len(image_files)])
    image_files = random.sample(image_files, nImages)
    logger.debug(f"Generating calibration image with {nImages} image files.")
    
    # Read the first image to get the dimensions
    first_image_path = os.path.join(directory, image_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = first_image.shape
    logger.debug(f"Calibration image will have size=({width}x{height}).")

    # Initialize an array to store all images
    all_images = np.zeros((len(image_files), height, width), dtype=np.uint8)

    # Load all images into the array
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        all_images[idx] = image

    # Compute the median image
    median_image = np.median(all_images, axis=0).astype(np.uint8)
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir, int(config['general']['dir_permissions']), exist_ok = True)
    #    logger.debug(f"Created new output directory for median image: {output_dir}.")
    cv2.imwrite(output_dir, median_image)
    logger.info(f"New median (calibration) image saved as {output_dir}.")

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)

    v_string = "V2024.10.04"

    logger = setup_logger('Shadowgraph Segmentation (main)', config)
    logger.info(f"Starting Shadowgraph segmentation script {v_string}")

    directory = sys.argv[1]
    if not os.path.exists(directory):
        logger.error(f'Specified path ({directory}) does not exist. Stopping.')
        sys.exit(1)

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
    # Subfolders of images(?)
    imgsets = []
    imgsets = [os.path.join(raw_dir, sub) for sub in next(os.walk(raw_dir))[1]]
    logger.info(f"Number of possible image sets found: {len(imgsets)}")
    for idx, f in enumerate(imgsets):
        logger.debug(f"Found ubfolder {idx}: {f}.")

    if len(imgsets) > 0:
        for im in imgsets:
            start_time = time()
            if not os.path.exists(config['segmentation']['calibration_image']):
                logger.info('Generating calibration image.')
                generate_median_image(im, config['segmentation']['calibration_image'])

        num_threads = min(os.cpu_count() - 2, len(imgsets))
        logger.info(f"Starting processing with {num_threads} processes...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            future_to_file = {executor.submit(process_image_dir, filename, segmentation_dir, config): filename for filename in imgsets}
        
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
                    logger.info(f"Processed {n} of {len(imgsets)} files.\t\t Estimated remainder: {(end_time - init_time)/n*(len(imgsets)-n) / 60:.1f} minutes.\t Elapsed time: {(end_time - init_time)/60:.1f} minutes.")
                    n += 1
    
        logger.info('Archiving results and cleaning up.')
        for im in imgsets:
            _, filename = os.path.split(im)
            output_path = segmentation_dir + os.path.sep + filename + os.path.sep
            logger.debug(f"Compressing to archive {filename + '.zip.'}")
            shutil.make_archive(segmentation_dir + os.path.sep + filename, 'zip', output_path)
            if not config['segmentation']['diagnostic']:
                logger.debug(f"Cleaning up output path: {output_path}.")
                delete_file(output_path, logger)
    
    logger.info(f"Finished segmentation. Total time: {(time() - init_time)/60:.1f} minutes.")
    sys.exit(0) # Successful close  
    


