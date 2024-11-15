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
import sqlite3
import threading

thread_local = threading.local()


def get_connection(db_name="data.db"):
    if not hasattr(thread_local, "connection"):
        thread_local.connection = sqlite3.connect(db_name, check_same_thread=False)
        thread_local.connection.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging for concurrency
    return thread_local.connection


def initialize_database():
    conn = get_connection()
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            n TEXT,
            i INTEGER,
            x INTEGER,
            y INTEGER,
            w INTEGER,
            h INTEGER,
            major_axis_length REAL,
            minor_axis_length REAL,
            area INTEGER,
            image BLOB
        )
        """)
    conn.close()

# [n, i, x, y, w, h, major_axis_length, minor_axis_length, area]

def insert_data(value):
    conn = get_connection()
    with conn:
        # Using transactions to ensure data integrity
        conn.executemany("INSERT INTO data (n, i, x, y, w, h, major_axis_length, minor_axis_length, area, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", value)
    #print(f"Data inserted: {value}")




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
    segmentation_dir = config['segmentation']['segmentation_dir']

    ## Find files to process:
    avis = findVideos(raw_dir, config, logger)
    
    timer['processing_start'] = time()
    ## Prepare workers for receiving frames
    num_threads = min(os.cpu_count() - 2, len(avis))
    logger.info(f"Starting processing with {num_threads} processes...")

    initialize_database()
    logger.info('Database initialized.')

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
    config = constructSegmentationDir(config)

    ## Run segmentation
    sidecar = mainSegmentation(config, logger)

    ## Write sidecar file
    json_save_pathname = config['segmentation']['segmentation_dir'] + '.json'
    json_object = json.dumps(sidecar, indent=4)
    with open(json_save_pathname, "w") as outfile:
        outfile.write(json_object)
        logger.debug("Sidecar writting finished.")

    sys.exit(0) # Successful close