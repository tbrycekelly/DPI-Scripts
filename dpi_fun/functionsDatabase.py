
import sqlite3
thread_local = threading.local()


def get_connection(config):
    if not hasattr(thread_local, "connection"):
        #if not os.path.exists(config['general']['database_path']):
        #    print('Database file does not exist. Was it initalized? Exiting immediately.')
        #    sys.exit(1)
        thread_local.connection = sqlite3.connect(config['general']['database_path'] + os.path.sep + config['identity'] + '.db', check_same_thread = False)
        thread_local.connection.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging for concurrency
    return thread_local.connection


def initialize_database(config, logger):
    try:
        os.makedirs(config['general']['database_path'], int(config['general']['dir_permissions']), exist_ok = True)
    except PermissionError:
        logger.error(f"Permission denied: Unable to create sqlite directory '{config['general']['database_path']}'.")
        sys.exit(1)
    except OSError as e:
        # Catch any other OS-related errors
        logger.error(f"Error creating directory '{config['general']['database_path']}': {e}")
        sys.exit(1)

    conn = get_connection(config)
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS segmentation (
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

        conn.execute("""
        CREATE TABLE IF NOT EXISTS classification (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            filename TEXT,
            frameNumber INTEGER,
            roiNumber INTEGER,
            classification BLOB
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
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
            prediction BLOB
        )
        """)

    conn.close()

def insert_data(value, config):
    conn = get_connection(config)
    with conn:
        conn.executemany("INSERT INTO segmentation (filename, frameNumber, roiNumber, x, y, w, h, major_axis_length, minor_axis_length, area, perimeter, min_gray_value, mean_gray_value, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", value)


def getROI(config):
    conn = get_connection(config)
    with conn:
        results = conn.query() #TODO

    return(results)

def insert_classifications(value, config):
    conn = get_connection(config)
    with conn:
        conn.executemany("INSERT INTO classification (model, filename, frameNumber, roiNumber, classification) VALUES (?, ?, ?, ?, ?)", value)

def insert_prediction(value, config):
    conn = get_connection(config)
    with conn:
        conn.executemany("INSERT INTO prediction (model, filename, frameNumber, roiNumber, prediction) VALUES (?, ?, ?, ?, ?)", value)
