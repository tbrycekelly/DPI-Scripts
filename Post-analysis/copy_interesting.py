import os
import shutil

def copy_large_files(source_dir, dest_dir, min_size_bytes):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    copied = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                if os.path.getsize(full_path) >= min_size_bytes:
                    dest_path = os.path.join(dest_dir, os.path.basename(full_path))
                    shutil.copy2(full_path, dest_path)
                    copied += 1
            except Exception as e:
                print(f"Error copying {full_path}: {e}")

    print(f"Copied {copied} files larger than {min_size_bytes / (1024):.2f} KB to '{dest_dir}'")


# === CONFIGURE THIS ===
source_directory = "/Volumes/T7 Shield/camera1/segmentation/2025-06-26 22-39-01.889-REG/"
destination_directory = "/Volumes/T7 Shield/SKQ202513S Interesting Critters2/"
minimum_size_kb = 100

# === RUN SCRIPT ===
copy_large_files(source_directory, destination_directory, minimum_size_kb * 1024)