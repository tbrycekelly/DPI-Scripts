#!/usr/bin/env python3
"""Classification script for DPI-Scripts

Usage:
    ./classification.py <dir>

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

import json
from dpi_fun import *

if __name__ == "__main__":
    """
    Entrypoint for running from command line.
    """
    if not os.path.exists('config.json'):
        print(f"Required configuration file 'config.json' not found. Aborting.")
        sys.exit(1)
    
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = setup_logger('Classification (main)', config)

    ## setup directories
    directory = sys.argv[-1]
    config['segmentation_dir'] = os.path.abspath(directory)
    config['classification_dir'] = config['segmentation_dir'].replace('segmentation', 'classification') + '-' + config["classification"]["model_name"]

    ## Run classification
    sidecar = mainClassifcation(config, logger)

    ## Write sidecar file
    json_save_pathname = config['classification_dir'] + '.json'
    json_object = json.dumps(sidecar, indent=4)
    with open(json_save_pathname, "w") as outfile:
        outfile.write(json_object)
        logger.debug("Sidecar writting finished.")
    
    sys.exit(0) # Successful close