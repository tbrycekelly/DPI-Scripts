import os
import sys
import shutil
import argparse
import glob
import logging
import configparser
import logging.config
import tqdm
import subprocess
import datetime
from time import time
from multiprocessing import Pool, Queue
import psutil
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import csv
from PIL import Image
import os
import pandas as pd
import json
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Process
import cv2


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