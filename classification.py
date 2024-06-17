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

## Source import.py
exec(open("./imports.py").read())

if __name__ == "__main__":
    
    with open('config.json', 'r') as f:
        config = json.load(f)

    v_string = "V2024.05.22"
    session_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).replace(':', '')
    logger = setup_logger('Classification (main)', config)
    logger.info(f"Starting Plankline Classification Script {v_string}")

    directory = sys.argv[1]
    if not os.path.exists(directory):
        logger.error(f'Specified path ({directory}) does not exist. Stopping.')
        sys.exit(1)


    if config['classification']['cpuonly']:
        logger.info("Disabling GPU support for this Tensorflow session.")
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
    

    # Load model
    model_path = f"../model/{config['classification']['model_name']}.keras"
    label_path = f"../model/{config['classification']['model_name']}.json"
    logger.info(f"Loading model from {model_path}.")
    logger.info(f"Loading model sidecar from {label_path}.")
    model = tf.keras.models.load_model(model_path)
    
    with open(label_path, 'r') as file:
        sidecar = json.load(file)
        logger.debug('Sidecar Loaded.')

    logger.info(f"Loaded keras model {config['classification']['model_name']} and sidecar JSON file.")
    
    # ### Setup Folders and run classification on each segment output
    segmentation_dir = os.path.abspath(directory)  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
    classification_dir = segmentation_dir.replace('segmentation', 'classification')  # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg
    classification_dir = classification_dir + '-' + config["classification"]["model_name"] # /media/plankline/Data/analysis/segmentation/Camera1/Transect1-reg-Plankton
    
    logger.debug(f"Segmentation directory is {segmentation_dir}.")
    logger.debug(f"Classification directory is {classification_dir}.")
    
    os.makedirs(classification_dir, int(config['general']['dir_permissions']), exist_ok = True)
    
    root = [z for z in os.listdir(segmentation_dir) if z.endswith('zip')]
    
    logger.info(f"Found {len(root)} archives for potential processing.")
    for idx, r in enumerate(root):
        logger.debug(f"Archive {idx}: {r}")

    n = 1
    init_time = time()
    for r in root:
        start_time = time()
        if not is_file_above_minimum_size(segmentation_dir + '/' + r, 128, logger):
            logger.warn(f"File {r} either does not exist or does not meet minimum size requirements (128 B). Skipping to next file.")
            continue

        ##Unpack zip file
        r2 = r.replace(".zip", "")

        classification_output_filepath = classification_dir + '/' + r2 + '_' + 'prediction.csv'
        if os.path.exists(classification_output_filepath):
            if config['classification']['overwrite']:
                logger.info(f"classification file exists for {r2} and overwrite is enabled. Overwritting prior classification file.")
                delete_file(classification_output_filepath, logger)
            else:
                logger.info(f"classification file exists for {r2}, overwrite is contraindicated in the config file. Skipping.")

        logger.debug(f"Unpacking archive at {segmentation_dir + '/' + r} to destination {segmentation_dir + '/' + r2}.")
        shutil.unpack_archive(segmentation_dir + '/' + r, segmentation_dir + '/' + r2 + "/", 'zip')

        ## Load and preprocess images
        images = []
        image_files = []
        for img in os.listdir(segmentation_dir + '/' + r2):
            if img.endswith(('png', 'jpeg', 'jpg', 'tif', 'tiff')): 
                logger.debug(f"Loading image {img}.")
                image_files.append(img)
                img = tf.keras.preprocessing.image.load_img(segmentation_dir + '/' + r2 + '/' + img,
                                                            target_size=(int(config['classification']['image_size']), int(config['classification']['image_size'])),
                                                            color_mode='grayscale')
                img = np.expand_dims(img, axis=0)
                images.append(img)
            else :
                logger.debug(f"Skipping file {img}.")
        images = np.vstack(images)
        
        logger.debug("Finished loading images. Starting prediction.")
        predictions = model.predict(images, verbose = 0)
        prediction_labels = np.argmax(predictions, axis=-1)
        prediction_labels = [sidecar['labels'][i] for i in prediction_labels]
        df = pd.DataFrame(predictions, index=image_files)
        df_short = pd.DataFrame(prediction_labels, index=image_files)
        
        
        df.columns = sidecar['labels']
        df.to_csv(classification_output_filepath, index=True, header=True, sep=',')
        df_short.columns = ['prediction']
        df_short.to_csv(classification_dir + '/' + r2 + '_' + 'predictionlist.csv', index=True, header=True, sep=',')
        
        logger.debug('Cleaning up unpacked archive files.')
        delete_file(segmentation_dir + '/' + r2 + "/", logger)
        end_time = time()
        logger.info(f"Processed {n} of {len(root)} files.\t\t Iteration: {end_time-start_time:.2f} seconds\t Estimated remainder: {(end_time - init_time)/n*(len(root)-n) / 60:.1f} minutes.\t Elapsed time: {(end_time - init_time)/60:.1f} minutes.")
        n+=1
    logger.info(f"Finished classification. Total time: {(end_time - init_time)/60:.1f} minutes.")
    sys.exit(0) # Successful close
