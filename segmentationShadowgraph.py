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
from functions import *
import platform
from functionsSegmentation import *


def process_frame(frame, config, logger):
    """
    This function processes each frame (provided as cv2 image frame) for flatfielding and segmentation. The steps include
    1. Flatfield intensities as indicated
    2. Segment the image using cv2 MSER algorithmn.
    3. Remove strongly overlapping bounding boxes
    4. Save cropped targets.
    """
    logger.debug(f"Pulled frame from queue. Processing {frame.get_name()} {frame.get_n()}.")
        
    image = np.array(cv2.imread(frame.read()))
    ## First: Apply calibration image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # targets = white, bkg = black

    image = image[776:2445,1155:3130]
    bkg = image
    imageSmooth = cv2.medianBlur(image, 5)
    bkg = cv2.medianBlur(bkg, 51)
    bkg = cv2.GaussianBlur(bkg, (151, 151), 0)
    gray = imageSmooth / bkg * 255
    gray = gray.clip(0,255).astype(np.uint8)
    grayAnnotated = gray

    #Third:  Apply Otsu's threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    name = frame.get_name()
    n = frame.get_n()
    logger.debug(f"Thresholding frame {n}.")
    stats = []
    minIntensity = 220

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
                mean_gray_value = np.mean(gray[y:(y+h), x:(x+w)])
                min_gray_value = np.mean(gray[y:(y+h), x:(x+w)])

                if 2*w + 2*h >= int(config['segmentation']['min_perimeter']) and 2*w + 2*h <= int(config['segmentation']['max_perimeter']) and min_gray_value <= minIntensity:
                    if config['segmentation']['diagnostic']:
                        cv2.rectangle(grayAnnotated, (x, y), (x+w, y+h), (0,0,255), 1)

                    size = max(w, h)
                    im = Image.fromarray(image[y:(y+h), x:(x+w)])
                    im_padded = Image.new(im.mode, (size, size), (255))
                    if (w > h):
                        left = 0
                        top = (size - h)//2
                    else:
                        left = (size - w)//2
                        top = 0
                    im_padded.paste(im, (left, top))
                    im_padded.save(f"{name}{n:06}-{i:06}.png")
                stats = [n, i, x + w/2, y + h/2, w, h, major_axis_length, minor_axis_length, area, min_gray_value, mean_gray_value]
                outwritter.writerow(stats)

        if config['segmentation']['diagnostic']:
            logger.debug(f"Diagnostic mode, saving threshold image and quantiledfiled image.")
            cv2.imwrite(f'{name}{n:06}-corrected.jpg', gray)
            cv2.imwrite(f'{name}{n:06}-annotated.jpg', grayAnnotated)
            cv2.imwrite(f'{name}{n:06}-original.jpg', image)
            cv2.imwrite(f'{name}{n:06}-background.jpg', bkg)
            cv2.imwrite(f'{name}{n:06}-threshold.jpg', thresh)
        logger.debug(f"Done with frame {n}.")
                


def process_image_dir(img_path, segmentation_dir, config):
    """
    This function will take an image folder as input and perform the following steps:
    1. Create output file structures/directories
    2. Load each frame, pass it through flatfielding and sequentially save segmented targets
    """
    logger = setup_logger('Segmentation Shadowgraph (Worker)', config)

    _, filename = os.path.split(img_path)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    os.makedirs(output_path, exist_ok=True)
    logger.debug(f"Created directory {output_path} if not already existing.")

    with open(f'{output_path[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
        logger.info(f"Initialized metrics file for {filename}.")
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        outwritter.writerow(['frame', 'roi', 'x', 'y', 'w', 'h', 'major_axis', 'minor_axis', 'area', 'min_grey_value', 'mean_grey_value'])

    logger.debug(f"Reading in calibration image {config['segmentation']['calibration_image']}.")

    for f in os.listdir(img_path):
      if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
          logger.debug(f"Processing image {f}.")
          process_frame(Frame(f, output_path, img_path + os.path.sep + f, 1, f), config, logger)
      else:
        logger.debug(f"Skipped reading non-image file {f}.") 



def generate_median_image(directory, output_dir):
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
    cv2.imwrite(output_dir, median_image)
    logger.info(f"New median (calibration) image saved as {output_dir}.")



def mainShadowgraphSegmentation(config, logger):
    """
    The main access function for segmentation. Can be called from external scripts, but designed initially
     to be called from __main__.
    """
    v_string = "V2024.10.14"
    logger.info(f"Starting segmentation script {v_string}")
    timer = {'init' : time()}

    if not os.path.exists(config['raw_dir']):
        logger.error(f'Specified path ({config["raw_dir"]}) does not exist. Stopping.')
        sys.exit(1)

    ## Determine directories
    raw_dir = config['raw_dir'] # /media/plankline/Data/raw/Camera0/test1
    segmentation_dir = config['segmentation_dir']

    ## Find files to process:
    imgsets = findImgsets(raw_dir, config, logger)
    
    timer['processing_start'] = time()
    ## Prepare workers for receiving frames
    num_threads = min(os.cpu_count() - 2, len(imgsets))
    logger.info(f"Starting processing with {num_threads} processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {executor.submit(process_image_dir, folder, segmentation_dir, config): folder for folder in imgsets}
    
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
    timer['processing_end'] = time()

    timer['archiving_start'] = time()
    logger.info('Archiving results and cleaning up.') # Important to isolate processing and cleanup since the threads don't know when everything is done processing.
    with concurrent.futures.ProcessPoolExecutor(max_workers = num_threads) as executor:
        future_to_file = {executor.submit(cleanupFile, filename, segmentation_dir, config): filename for filename in imgsets}
    
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
                logger.info(f"Cleaning up {n} of {len(imgsets)} files.\t\t Estimated remainder: {(end_time - init_time2)/n*(len(imgsets)-n) / 60:.1f} minutes.\t Elapsed time: {(end_time - init_time2)/60:.1f} minutes.")
                n += 1

    timer['archiving_end'] = time()
    logger.info(f"Finished segmentation. Total time: {(time() - init_time)/60:.1f} minutes.")

    sidecar = {
        'directory' : directory,
        'nFiles' : len(imgsets),
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

    return sidecar


if __name__ == "__main__":

    if not os.path.exists('config.json'):
        print(f"Required configuration file 'config.json' not found. Aborting.")
        sys.exit(1)

    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = setup_logger('Shadowgraph Segmentation (main)', config)

    directory = sys.argv[1]
    config['raw_dir'] = os.path.abspath(directory)
    config['segmentation_dir'] = constructSegmentationDir(config['raw_dir'], config)

    ## Run segmentation
    sidecar = mainShadowgraphSegmentation(config, logger)

    ## Write sidecar file
    json_save_pathname = config['segmentation_dir'] + '.json'
    json_object = json.dumps(sidecar, indent=4)
    with open(json_save_pathname, "w") as outfile:
        outfile.write(json_object)
        logger.debug("Sidecar writting finished.")

    sys.exit(0) # Successful close

    


