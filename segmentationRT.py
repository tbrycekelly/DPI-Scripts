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

import json
from dpi_fun import *
#from dpi_fun.functions import *
#from dpi_fun.functionsSegmentation import *

if __name__ == "__main__":
    """
    Entrypoint for script when run from the command line.
    """
    if not os.path.exists('configRT.json'):
        print(f"Required configuration file 'config.json' not found. Aborting.")
        sys.exit(1)
    
    with open('configRT.json', 'r') as f:
        config = json.load(f)

    logger = setup_logger('Segmentation (Main)', config)

    ## Setup directories
    directory = sys.argv[-1]
    config['raw_dir'] = os.path.abspath(directory)
    config['segmentation_dir'] = constructSegmentationDir(config['raw_dir'], config)
    _, filename = os.path.split(config['raw_dir'])
    config['identity'] = filename

    seenFiles = set()

    while True:
        ## Find files to process:
        avis = findVideos(config, logger, checkWriteStatus = True)
        imgsets = findImgsets(config, logger)

        avisNew = set(avis).difference(seenFiles)
        imgsetsNew = set(imgsets).difference(seenFiles)

        N = len(avisNew) + len(imgsetsNew)

        if N > 0:
            ## Run segmentation
            start_time = time()
            sidecar = mainSegmentation(config, logger, avisNew, imgsetsNew)
            end_time = time()
            logger.info(f"Processed {N} new AVI files.\t Elapsed time: {(end_time - start_time):.1f} seconds.")

            ## Write sidecar file
            json_save_pathname = config['segmentation_dir'] + sidecar['sessionid'] + '.json'
            json_object = json.dumps(sidecar, indent=4)
            with open(json_save_pathname, "w") as outfile:
                outfile.write(json_object)
                logger.debug("Sidecar writting finished.")
            seenFiles = avisNew.union(imgsetsNew).union(seenFiles)
        else:
            logger.info("No new files discovered.")
        sleep(int(config['segmentation']['wait_time']))

    sys.exit(0) # Successful close