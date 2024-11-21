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
import sqlite3
import threading

thread_local = threading.local()

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
    timer['processing_end'] = time()

    timer['archiving_start'] = time()
    if 'csv' in config['general']['export_as']:
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



def mainShadowgraphSegmentation(config, logger):
    """
    The main access function for segmentation. Can be called from external scripts, but designed initially
     to be called from __main__.
    """
    v_string = "V2024.10.14"
    logger.info(f"Starting segmentation script {v_string}")
    timer = {'init' : time()}

    ##TODO  Test that config is valid: raw_dir, segmentation_dir...?

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
    if 'csv' in config['general']['export_as']:
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
    statistics_filepath = output_path[:-1] + ' statistics.csv'
    
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
        outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
        outwritter.writerow(['frame', 'roi', *calcStats()]) # Names: frame, roi, ...

    # Open video file, initialize statistcs file, and start going through frames.
    video = cv2.VideoCapture(avi_path)
    if not video.isOpened():
        logger.error(f"Issue openning video {avi_path}.")
        return

    n = 1 # Frame count
    while True:
        ret, frame = video.read()
        if ret:
            if not frame is None:
                process_linescan_frame(Frame(sourcePath = avi_path, destPath = output_path, filename = filename, frameNumber = n, data = frame), config, logger)
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

    if 'csv' in config['general']['export_as']:
        try:
            os.makedirs(output_path, exist_ok=True)
            logger.debug(f"Created directory {output_path} if not already existing.")
        except PermissionError:
            logger.error(f"Permission denied: Unable to create output directory '{output_path}'.")
            return
        except OSError as e:
            # Catch any other OS-related errors
            logger.error(f"Error creating directory '{output_path}': {e}")

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
            logger.info(f"Initialized metrics file for {filename}.")
            outwritter = csv.writer(outcsv, delimiter=',', quotechar='|')
            outwritter.writerow(['frame', 'roi', *calcStats()])

    logger.debug(f"Reading in calibration image {config['segmentation']['calibration_image']}.")
    k = 1

    for f in os.listdir(img_path):    
        if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            logger.debug(f"Processing image {f}.")
            process_area_frame(Frame(sourcePath = img_path + os.path.sep, destPath = output_path, filename = f, frameNumber = k, data = None), config, logger)
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

            if 2*w + 2*h > int(config['segmentation']['min_perimeter_statsonly']):
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
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-qualtilefield.jpg', gray)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-annotated.jpg', grayAnnotated)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-threshold.jpg', thresh)


def process_area_frame(frame, config, logger):
    """
    This function processes each frame (provided as cv2 image frame) for flatfielding and segmentation. The steps include
    1. Flatfield intensities as indicated
    2. Segment the image using cv2 MSER algorithmn.
    3. Remove strongly overlapping bounding boxes
    4. Save cropped targets.
    """
    logger.debug(f"Pulled frame from queue. Processing {frame.get_filename()} {frame.get_frame_number()}.")
        
    image = cv2.imread(frame.get_source_path() + frame.get_filename())
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Center of the image
    radius = 1100
    #image = image[776:2445,1155:3130]
    image = image[(center[1]-radius):(center[1] + radius), (center[0]-radius):(center[0] + radius)]
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Center of the image
    mask = np.zeros(image.shape[:2], dtype="uint8")
    # Example radius
    cv2.circle(mask, center, radius, 255, -1)
    image = ~cv2.bitwise_and(~image, ~image, mask=mask)

    resize = 1
    #image = cv2.resize(image, (image.shape[1] // resize, image.shape[0] // resize))
    image = np.array(image)

    ## First: Apply calibration image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # targets = white, bkg = black
    imageSmooth = cv2.medianBlur(image, 1//resize)
    bkg = cv2.medianBlur(image, 50//resize + 1)
    bkg = cv2.GaussianBlur(bkg, (150 // resize + 1, 150 // resize + 1), 0)
    gray = (imageSmooth / (bkg + 1/255)) * 255
    gray = gray.clip(0,255).astype(np.uint8)
    grayAnnotated = gray

    #Third:  Apply Otsu's threshold
    thresh = calcThreshold(gray)
    edges = cv2.Canny(gray, 10, 150, L2gradient = True)
    #thresh = cv2.bitwise_or(thresh, edges)
    
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

            if 2*w + 2*h > int(config['segmentation']['min_perimeter_statsonly']):
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
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-annotated.jpg', grayAnnotated)
        #cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-original.jpg', image)
        #cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-background.jpg', bkg)
        cv2.imwrite(f'{destPath[:-1] + " diagnostic" + os.path.sep}{fileName}-{frameNumber:06}-threshold.jpg', thresh)
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
    logger = setup_logger('Segmentation (Main)', config)
    logger.debug(f"Segmentation directory: {segmentationDir}")
    try:
        os.makedirs(segmentationDir, int(config['general']['dir_permissions']), exist_ok = True)
    except PermissionError:
        logger.error(f"Permission denied: Unable to create segmentation directory '{directory_path}'.")
        sys.exit(1)
    except OSError as e:
        # Catch any other OS-related errors
        logger.error(f"Error creating directory '{directory_path}': {e}")

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


def findImgsets(raw_dir, config, logger):
    """
    Helper function to search for availabile video files in a directory.
    """
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


def calcThreshold(gray):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return(thresh)


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
    Helper function to construct a new logger.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        return logger
    
    logger = logging.getLogger(name) 
    logger.setLevel(logging.DEBUG) # the level should be the lowest level set in handlers

    log_format = logging.Formatter('[%(levelname)s] (%(process)d) %(asctime)s - %(message)s')
    if not os.path.exists(config['general']['log_path']):
        try:
            os.makedirs(config['general']['log_path'])
        except PermissionError:
            print(f"Permission denied: Unable to create directory '{config['general']['log_path']}'.")
            print('Logging will not be performed and may crash the script.')
        except OSError as e:
            print(f"Error creating directory '{config['general']['log_path']}': {e}")
            print('Logging will not be performed and may crash the script.')

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



def get_connection(db_name="data.db"):
    if not hasattr(thread_local, "connection"):
        thread_local.connection = sqlite3.connect(db_name, check_same_thread=False)
        thread_local.connection.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging for concurrency
    return thread_local.connection


def initialize_database(config):
    conn = get_connection(config['db_path'])
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            frameNumber INTEGER,
            roiNumber INTEGER,
            x INTEGER,
            y INTEGER,
            w INTEGER,
            h INTEGER,
            major_axis_length REAL,
            minor_axis_length REAL,
            area INTEGER,
            perimeter INTEGER,
            min_gray_value INTEGER,
            mean_gray_value INTEGER,
            image BLOB
        )
        """)
    conn.close()

def insert_data(value, config):
    conn = get_connection(config['db_path'])
    with conn:
        # Using transactions to ensure data integrity
        conn.executemany("INSERT INTO data (filename, frameNumber, roiNumber, x, y, w, h, major_axis_length, minor_axis_length, area, perimeter, min_gray_value, mean_gray_value, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", value)
    #print(f"Data inserted: {value}")

