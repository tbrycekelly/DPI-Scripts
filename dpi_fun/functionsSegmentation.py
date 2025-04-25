import logging
import logging.config
import os
import shutil
from logging.handlers import TimedRotatingFileHandler
from time import time
import sys
import numpy as np
import csv
from PIL import Image
import json
import cv2
import threading
import concurrent.futures
import platform
from .functions import *


class Frame:
    def __init__(self, sourcePath, destPath, filename, frameNumber, data):
        self.sourcePath = sourcePath # Path to raw video/imagefolder
        self.destPath = destPath # Path to raw video/imagefolder
        self.filename = filename # name of video or folder
        self.frameNumber = frameNumber # frame ID
        self.data = data # frame data

    def get_frame_number(self):
        return (self.frameNumber)

    def get_source_path(self):
        return self.sourcePath

    def get_dest_path(self):
        return self.destPath
    
    def get_filename(self):
        return self.filename

    def read(self):
        return self.data
    
    def update(self, newframe):
        self.frame = newframe


def mainSegmentation(config, logger, avis, imgsets):
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
    
    if 'sqlite' in config['general']['export_as']:
        # SQLite Init
        if not os.path.exists(config['general']['database_path']):
            logger.info(f"Database path does not exist, creating now: {config['general']['database_path']}")
            try:
                os.makedirs(config['general']['database_path'], mode = config['general']['dir_permissions'],  exist_ok = True) # TODO dir_permissiosn throughout.
            except PermissionError:
                logger.error(f"Permission denied while attempting to make database directory: {config['segmentation']['scratch_dir']}.") # TODO Further exception.
                sys.exit(1)

        config['db_path'] = config['general']['database_path'] + os.path.sep + 'database.db'
        initialize_database(config, logger)
        logger.info('Database initialized.')

    ## Determine directories
    raw_dir = config['raw_dir'] # /media/plankline/Data/raw/Camera0/test1
    segmentation_dir = config['segmentation_dir']

    if config['segmentation']['use_scratch']:
        try:
            os.makedirs(config['segmentation']['scratch_dir'], mode = config['general']['dir_permissions'],  exist_ok = True) # TODO dir_permissiosn throughout.
        except PermissionError:
            logger.error(f"Permission denied while attempting to make scratch directory: {config['segmentation']['scratch_dir']}.") # TODO Further exception.
    
    timer['processing_start'] = time()
    ## Prepare workers for receiving frames
    num_threads = min(os.cpu_count() - 2, max(len(avis), len(imgsets)))
    logger.info(f"Starting processing with {num_threads} processes...")

    if len(avis) > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            future_to_file = {executor.submit(process_video, segmentation_dir, config, filename): filename for filename in avis}
        
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
    
    if len(imgsets) > 0:
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
    if 'csv' in config['general']['export_as'] and len(avis) > 0 and config['segmentation']['cleanup']:
        logger.info('Archiving results and cleaning up.') # Important to isolate processing and cleanup since the threads don't know when everything is done processing.
        with concurrent.futures.ProcessPoolExecutor(max_workers = num_threads) as executor:
            future_to_file = {executor.submit(cleanupFile, filename, segmentation_dir, config): filename for filename in avis}
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

    if 'csv' in config['general']['export_as'] and len(imgsets) > 0 and config['segmentation']['cleanup']:
        logger.info('Archiving results and cleaning up.') # Important to isolate processing and cleanup since the threads don't know when everything is done processing.
        with concurrent.futures.ProcessPoolExecutor(max_workers = num_threads) as executor:
            future_to_file = {executor.submit(cleanupFile, filename, segmentation_dir, config): filename for filename in imgsets}
        
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
        'nFiles' : len(avis) + len(imgsets),
        'script_version' : v_string,
        'sessionid': str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', ''),
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


def process_video(segmentation_dir, config, avi_path):
    """
    This function will take an avi filepath as input and perform the following steps:
    1. Create output file structures/directories
    2. Load each frame, pass it through flatfielding and sequentially save segmented targets
    """
    # segmentation_dir: /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    logger = setup_logger('Segmentation (Worker)', config)
    _, filename = os.path.split(avi_path)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    scratch_path = get_scratch(config, filename)
    statistics_filepath = scratch_path[:-1] + ' statistics.csv'
    
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

    if 'csv' in config['general']['export_as']:
        try:
            os.makedirs(scratch_path, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied when making directory {scratch_path}.")

        try:
            os.makedirs(output_path, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied when making directory {output_path}.")

        if config['segmentation']['diagnostic']:
            diagnostic_path = segmentation_dir + os.path.sep + filename + " diagnostic" + os.path.sep
            try:
                os.makedirs(diagnostic_path, exist_ok=True)
            except PermissionError:
                logger.error(f"Permission denied when making diagnostic directory {diagnostic_path}.")

    with open(statistics_filepath, 'a', newline='\n') as outcsv:
        logger.debug(f"Initialized metrics file for {statistics_filepath}.")
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        outwritter.writerow(['frame', 'roi', *calcStats()]) # Names: frame, roi, ...

    # Open video file, initialize statistcs file, and start going through frames.
    video = cv2.VideoCapture(avi_path)
    if not video.isOpened():
        logger.error(f"Issue openning video {avi_path}.")
        return

    n = 1 # Frame count
    while True:
        good_return, frame = video.read()
        if good_return:
            if not frame is None:
                process_linescan_frame(Frame(sourcePath = avi_path, destPath = scratch_path, filename = filename, frameNumber = n, data = frame), config, logger)
                n += 1 # Increment frame counter.
        else:
            break


def process_image_dir(img_path, segmentation_dir, config):
    """
    This function will take an image folder as input and perform the following steps:
    1. Create output file structures/directories
    2. Load each frame, pass it through flatfielding and sequentially save segmented targets
    """
    logger = setup_logger('Segmentation Shadowgraph (Worker)', config)

    _, filename = os.path.split(img_path)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    scratch_path = get_scratch(config, filename)

    if 'csv' in config['general']['export_as']: # Need to construct some directories
        try:
            os.makedirs(output_path, exist_ok=True)
            logger.debug(f"Created directory {output_path} if not already existing.")
        except PermissionError:
            logger.error(f"Permission denied: Unable to create output directory '{output_path}'.")
            return
        except OSError as e:
            # Catch any other OS-related errors
            logger.error(f"Error creating directory '{output_path}': {e}")

        try:
            os.makedirs(scratch_path, exist_ok=True)
            logger.debug(f"Created directory {scratch_path} if not already existing.")
        except PermissionError:
            logger.error(f"Permission denied: Unable to create scratch directory '{scratch_path}'.")
            return
        except OSError as e:
            # Catch any other OS-related errors
            logger.error(f"Error creating directory '{scratch_path}': {e}")

        if config['segmentation']['diagnostic']:
            try:
                diagnostic_path = segmentation_dir + os.path.sep + filename + " diagnostic" + os.path.sep
                os.makedirs(diagnostic_path, exist_ok=True)
            except PermissionError:
                logger.error(f"Permission denied: Unable to create diagnostic directory '{diagnostic_path}'.")
            except OSError as e:
                # Catch any other OS-related errors
                logger.error(f"Error creating directory '{diagnostic_path}': {e}")

        with open(f'{output_path[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
            logger.debug(f"Initialized metrics file for {filename}.")
            outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
            outwritter.writerow(['frame', 'roi', *calcStats()])

    valid_extensions = tuple(config['segmentation']['image_extensions'])
    logger.debug(f"Valid extensions: {', '.join(valid_extensions)}")

    k = 1
    for f in os.listdir(img_path):    
        if f.endswith(valid_extensions):
            logger.debug(f"Processing image {f}.")
            process_area_frame(
                Frame(sourcePath = img_path + os.path.sep, destPath = scratch_path, filename = f,frameNumber = k, data = None),
                 config, logger, apply_corrections=True
                )
            k = k + 1
        else:
            logger.debug(f"Skipped reading non-image file {f}.") 


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
    logger.debug(f"Pulled frame from queue. Processing {frame.get_filename()}.")
        
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
    fileName = frame.get_filename()
    frameNumber = frame.get_frame_number()
    
    stats = []
    
    # Open statistics file and iterate through all identified ROIs.
    with open(f'{destPath[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
        logger.debug(f"Writing to statistics.csv. Found {len(cnts)} ROIs.")
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        stats = []
        for i in range(len(cnts)):
            x,y,w,h = cv2.boundingRect(cnts[i])

            if 2*w + 2*h >= int(config['segmentation']['min_perimeter']) and 2*w + 2*h <= int(config['segmentation']['max_perimeter']):
                if 'sqlite' in config['general']['export_as']:
                    roiblob = gray[y:(y+h), x:(x+w)].tobytes()
                if 'csv' in config['general']['export_as']:
                    saveROI(f"{destPath}{fileName}-{frameNumber:06}-{i:06}.png", gray[y:(y+h), x:(x+w)], w, h)
                if config['segmentation']['diagnostic']:
                    cv2.rectangle(grayAnnotated, (x, y), (x+w, y+h), (0,0,255), 1)
            else:
                roiblob = np.zeros(0).tobytes()

            if 2*w + 2*h >= int(config['segmentation']['min_perimeter_statsonly']):
                newstat = calcStats(gray[y:(y+h), x:(x+w)], cnts[i], x, y, w, h)
                if 'sqlite' in config['general']['export_as']:
                    stats.append([fileName, frameNumber, i, *newstat, sqlite3.Binary(roiblob)])
                if 'csv' in config['general']['export_as']:
                    outwritter.writerow([frameNumber, i, *newstat])

        if 'sqlite' in config['general']['export_as']:
            insert_data(stats, config)

    # Save optional diagnsotic images before returning.
    if config['segmentation']['diagnostic']:
        logger.debug('Saving diagnostic images.')
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-3qualtilefield.jpg', gray)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-4annotated.jpg', grayAnnotated)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-2threshold.jpg', thresh)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-1original.jpg', frame.read())


def process_area_frame(frame, config, logger, apply_corrections = True):
    """
    This function processes each frame (provided as cv2 image frame) for flatfielding and segmentation. The steps include
    1. Flatfield intensities as indicated
    2. Segment the image using cv2 MSER algorithmn.
    3. Remove strongly overlapping bounding boxes
    4. Save cropped targets.
    """
    logger.debug(f"Pulled frame from queue. Processing {frame.get_filename()} {frame.get_frame_number()}.")
        
    image = ~cv2.imread(frame.get_source_path() + frame.get_filename(), 0) # white = target, black = bkg
    
    if apply_corrections:
        center = (image.shape[1] // 2, image.shape[0] // 2)  # Center of the image
        radius = 1050

        image = image[(center[1]-radius):(center[1] + radius), (center[0]-radius):(center[0] + radius)]
        center = (image.shape[1] // 2, image.shape[0] // 2)  # Center of the image
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, center, radius, 255, -1)
        image = ~cv2.bitwise_and(image, image, mask=mask)

        resize = 1
        #image = cv2.resize(image, (image.shape[1] // resize, image.shape[0] // resize))
        
        # Rescale to unit brightness
        image = np.array(image)  
        offset = np.min(image)
        f = 255 / np.median(image - offset)
        image = (image - offset) * f
        
        # TODO Improve this area
        image = image.clip(0,255).astype(np.uint8)
        imageSmooth = cv2.medianBlur(image, 1//resize)
        bkg = cv2.medianBlur(image, 150//resize + 1)
        bkg = cv2.GaussianBlur(bkg, (150 // resize + 1, 150 // resize + 1), 0)
        gray = (imageSmooth / (bkg + 1/255)) * 255
        gray = gray.clip(0,255).astype(np.uint8)
        
    else:
        gray = image.copy()
        bkg = np.zeros(image.shape[:2], dtype="uint8")

    grayAnnotated = gray.copy()
    grayAnnotated = cv2.cvtColor(grayAnnotated, cv2.COLOR_GRAY2RGB)

    #Third:  Apply Otsu's threshold
    thresh = calcThreshold(gray)
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    destPath = frame.get_dest_path()
    fileName = frame.get_filename()
    frameNumber = frame.get_frame_number()
    logger.debug(f"Thresholding frame {frameNumber}.")
    stats = []

    with open(f'{destPath[:-1]} statistics.csv', 'a', newline='\n') as outcsv:
        logger.debug(f"Writing to statistics.csv. Found {len(cnts)} ROIs.")
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        stats = []
        for i in range(len(cnts)):
            x,y,w,h = cv2.boundingRect(cnts[i])

            if 2*w + 2*h >= int(config['segmentation']['min_perimeter']) and 2*w + 2*h <= int(config['segmentation']['max_perimeter']):
                if 'sqlite' in config['general']['export_as']:
                    roiblob = gray[y:(y+h), x:(x+w)].tobytes()
                if 'csv' in config['general']['export_as']:
                    saveROI(f"{destPath}{fileName}-{frameNumber:06}-{i:06}.png", gray[y:(y+h), x:(x+w)], w, h)
                if config['segmentation']['diagnostic']:
                    cv2.rectangle(grayAnnotated, (x, y), (x+w, y+h), (0,0,255), 1)
            else:
                roiblob = np.zeros(0).tobytes()

            if 2*w + 2*h >= int(config['segmentation']['min_perimeter_statsonly']):
                newstat = calcStats(gray[y:(y+h), x:(x+w)], cnts[i], x, y, w, h)
                if 'sqlite' in config['general']['export_as']:
                    stats.append([fileName, frameNumber, i, *newstat, sqlite3.Binary(roiblob)])
                if 'csv' in config['general']['export_as']:
                    outwritter.writerow([frameNumber, i, *newstat])

        if 'sqlite' in config['general']['export_as']:
            insert_data(stats, config)
                
    if config['segmentation']['diagnostic']:
        logger.debug(f"Diagnostic mode, saving threshold image and quantiledfiled image.")
        #cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-corrected.jpg', gray)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-2annotated.jpg', grayAnnotated)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-1original.jpg', image)
        #cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-background.jpg', bkg)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-3threshold.jpg', thresh)
    logger.debug(f"Done with frame {frameNumber}.")                


def cleanupFile(rawPath, segmentation_dir, config):
    """
    cleanupFile: Standalone function to move a segmentated directory of ROIs into a compressed
    zip archive. It then deletes the folder if not in diagnostic mode. This function allows for
    multithreaded operation.
    """
    logger = setup_logger('Segmentation (Worker)', config)
    _, filename = os.path.split(rawPath)
    output_path = segmentation_dir + os.path.sep + filename + os.path.sep
    scratch_path = get_scratch(config, filename)

    logger.debug(f"Compressing to archive {filename + '.zip.'}")
    if is_file_above_minimum_size(segmentation_dir + os.path.sep + filename + '.zip', 0, logger):
        if config['segmentation']['overwrite']:
            logger.warn(f"archive exists for {filename} and overwritting is allowed. Deleting old archive.")
            delete_file(segmentation_dir + os.path.sep + filename + '.zip', logger)
        else:
            logger.warn(f"archive exists for {filename} and overwritting is allowed. Skipping Archiving")
    
    shutil.move(scratch_path[:-1] + ' statistics.csv', output_path) # TODO
    delete_file(scratch_path[:-1] + ' statistics.csv', logger) #TODO

    shutil.make_archive(segmentation_dir + os.path.sep + filename, 'zip', scratch_path)
    if not config['segmentation']['diagnostic']:
        logger.debug(f"Cleaning up output path: {output_path}.")
        delete_file(scratch_path, logger)


def constructSegmentationDir(rawPath, config):
    """
    Helper function to standardize the directory construction for the segmentation output given a configuration
    and raw file path.
    """
    segmentationDir = rawPath.replace("raw", "analysis") # /media/plankline/Data/analysis/Camera1/Transect1

    for i in range(5):
        segmentationDir = segmentationDir.replace(f"camera{i}/", f"camera{i}/segmentation/") # /media/plankline/Data/analysis/Camera1/Transect1
        
    segmentationDir = segmentationDir + f"-{config['segmentation']['basename']}" # /media/plankline/Data/analysis/segmentation/Camera1/segmentation/Transect1-REG
    logger = setup_logger('Segmentation (Main)', config)
    logger.debug(f"Segmentation directory: {segmentationDir}")
    try:
        os.makedirs(segmentationDir, int(config['general']['dir_permissions']), exist_ok = True)
    except PermissionError:
        logger.error(f"Permission denied: Unable to create segmentation directory '{directory_path}'.")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Error creating directory '{directory_path}': {e}")

    return(segmentationDir)


def findVideos(config, logger, checkWriteStatus = False):
    """
    Helper function to search for availabile video files in a directory.
    """
    raw_dir = config['raw_dir']
    avis = []
    valid_extensions = tuple(config['segmentation']['video_extensions'])
    logger.debug(f"Valid extensions: {', '.join(valid_extensions)}")

    if checkWriteStatus:
        avis = [os.path.join(raw_dir, avi) for avi in os.listdir(raw_dir) if avi.endswith(valid_extensions) and is_file_finished_writing(os.path.join(raw_dir, avi))]
    else:
        avis = [os.path.join(raw_dir, avi) for avi in os.listdir(raw_dir) if avi.endswith(valid_extensions)]
        
    logger.info(f"Number of videos found: {len(avis)}")

    for idx, av in enumerate(avis):
        logger.debug(f"Found video file {idx}: {av}.")

    return(avis)


def findImgsets(config, logger, checkWriteStatus = False):
    """
    Helper function to search for availabile video files in a directory.
    """
    raw_dir = config['raw_dir']
    imgsets = []
    imgsets = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    logger.info(f"Number of imgsets found: {len(imgsets)}")

    for idx, av in enumerate(imgsets):
        logger.debug(f"Found imgsets folder {idx}: {av}.")

    return(imgsets)


def calcStats(grayROI = None, cnt = None, x = None, y = None, w = None, h = None):
    if cnt is None:
        return(['x', 'y', 'w', 'h', 'major_axis', 'minor_axis', 'area', 'perimeter', 'min_gray_value', 'mean_gray_value'])

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if len(cnt) >= 5:  # Minimum number of points required to fit an ellipse
        ellipse = cv2.fitEllipse(cnt)
        center, axes, angle = ellipse
        major_axis_length = round(max(axes),1)
        minor_axis_length = round(min(axes),1)
    else :
        major_axis_length = -1
        minor_axis_length = -1
    mean_gray_value = np.mean(grayROI)
    min_gray_value = np.mean(grayROI)
    return([x, y, w, h, major_axis_length, minor_axis_length, int(area), int(perimeter), int(min_gray_value), int(mean_gray_value)])


def saveROI(filename, imagedata, w, h):
    size = max(w, h)
    im = Image.fromarray(imagedata)
    im_padded = Image.new(im.mode, (size, size), (255))
    if (w > h):
        left = 0
        top = (size - h)//2
    else:
        left = (size - w)//2
        top = 0
    im_padded.paste(im, (left, top))
    im_padded.save(filename)
    return(0)


def calcThreshold(gray, runCanny = True, cannyParams = (30,80), dilateKernel = (5,5)):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    if runCanny: # Via Mark Y!
        graySmooth = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(graySmooth, cannyParams[0], cannyParams[1], L2gradient = True)
        im_out_canny = np.ones_like(gray) * 255
        im_out_canny = cv2.bitwise_and(im_out_canny, edges)
        thresh = thresh | im_out_canny

    # dilate to connect edges for flood fill
    kernel = np.ones(dilateKernel, np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return(thresh)

def get_scratch(config, filename):
    if config['segmentation']['use_scratch']:
        return(config['segmentation']['scratch_dir'] + os.path.sep + filename + os.path.sep)
    return(config['segmentation_dir'] + os.path.sep + filename + os.path.sep)

