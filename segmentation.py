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
from functions import *
import platform
from functionsSegmentation import *


def process_linescan_frame(frame, config, logger):
    """
    Function for processing a single frame object:
        Flatfielding
        Thresholding
        Contouring
        Segmentation
        Statistics
        ROIs
    """
    logger.debug(f"Pulled frame from queue. Processing {frame.get_name()}.")
        
    ## Read img and flatfield
    gray = cv2.cvtColor(frame.read(), cv2.COLOR_BGR2GRAY)
    gray = np.array(gray)
    
    field = np.quantile(gray, q = float(config['segmentation']['flatfield_q']), axis = 0)
    gray = (gray / field.T * 255.0)
    gray = gray.clip(0,255).astype(np.uint8)
    grayAnnotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Apply Otsu's threshold
    thresh = calcThreshold(gray)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
    destPath = frame.get_dest_path() # filepath to the movie or imageset
    filename = frame.get_filename()
    frameNumber = frame.get_frame_number()
    
    stats = []
    
    # Open statistics file and iterate through all identified ROIs.
    with open(f'{destPath[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
        logger.debug(f"Writing to statistics.csv. Found {len(cnts)} ROIs.")
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        for i in range(len(cnts)):
            x,y,w,h = cv2.boundingRect(cnts[i])

            # If ROI is of useful minimum size.
            if 2*w + 2*h >= int(config['segmentation']['min_perimeter_statsonly']):
                stats = [frameNumber, i, *calcStats(cnts[i], x, y, w, h)]
                outwritter.writerow(stats)

            # If ROI is within size limits for saving an image. 
            if 2*w + 2*h >= int(config['segmentation']['min_perimeter']) and 2*w + 2*h <= int(config['segmentation']['max_perimeter']):
                saveROI(f"{destPath}{filename}-{frameNumber:06}-{i:06}.png", gray[y:(y+h), x:(x+w)], w, h) 
                if config['segmentation']['diagnostic']:
                    cv2.rectangle(grayAnnotated, (x, y), (x+w, y+h), (0,0,255), 1)

    # Save optional diagnsotic images before returning.
    if config['segmentation']['diagnostic']:
        logger.debug('Saving diagnostic images.')
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{filename}-{frameNumber:06}-qualtilefield.jpg', gray)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{filename}-{frameNumber:06}-annotated.jpg', grayAnnotated)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{filename}-{frameNumber:06}-threshold.jpg', thresh)


def mainSegmentation(config, logger):
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
    avis = findVideos(raw_dir, config, logger)
    
    timer['processing_start'] = time()
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
    timer['processing_end'] = time()

    timer['archiving_start'] = time()
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

    timer['archiving_end'] = time()
    logger.info(f"Finished segmentation. Total time: {(time() - init_time)/60:.1f} minutes.")

    sidecar = {
        'directory' : directory,
        'nFiles' : len(avis),
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
    """
    Entrypoint for script when run from the command line.
    """
    if not os.path.exists('config.json'):
        print(f"Required configuration file 'config.json' not found. Aborting.")
        sys.exit(1)
    
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = setup_logger('Segmentation (Main)', config)

    ## Setup directories
    directory = sys.argv[1]
    config['raw_dir'] = os.path.abspath(directory)
    config['segmentation_dir'] = constructSegmentationDir(config['raw_dir'], config)

    ## Run segmentation
    sidecar = mainSegmentation(config, logger)

    ## Write sidecar file
    json_save_pathname = config['segmentation_dir'] + '.json'
    json_object = json.dumps(sidecar, indent=4)
    with open(json_save_pathname, "w") as outfile:
        outfile.write(json_object)
        logger.debug("Sidecar writting finished.")

    sys.exit(0) # Successful close