import os
import shutil

# --- CONFIGURATION ---
# Path to the downloaded dataset folder (contains 00, 01, ... 09)
RAW_DATA_PATH = "leapGestRecog" 

# Where we want to put the clean data
OUTPUT_PATH = "data"

# Which gestures do you want to keep? (Mapping original name -> new name)
# You can comment out lines to ignore gestures you don't need.
GESTURES_TO_KEEP = {
    "01_palm": "palm",
    "03_fist": "fist",
    "05_thumb": "thumb",
    "06_index": "index",
    "07_ok": "ok",
    # "02_l": "l_sign",
    # "10_down": "down",
}

def organize_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Could not find '{RAW_DATA_PATH}' folder.")
        print("Please download the dataset and unzip it next to this script.")
        return

    # Create the main output directory
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Loop through all Subject folders (00, 01, ... 09)
    # The dataset is split by 'subject' (person), but we want to merge them.
    for subject_folder in os.listdir(RAW_DATA_PATH):
        subject_path = os.path.join(RAW_DATA_PATH, subject_folder)
        
        if not os.path.isdir(subject_path):
            continue

        print(f"Processing Subject: {subject_folder}...")

        # Loop through the gestures inside this subject (01_palm, 02_l, etc.)
        for gesture_folder in os.listdir(subject_path):
            
            # Skip gestures we didn't list in GESTURES_TO_KEEP
            if gesture_folder not in GESTURES_TO_KEEP:
                continue
                
            new_name = GESTURES_TO_KEEP[gesture_folder]
            
            # Create the specific gesture folder in our output path (e.g., data/fist)
            target_folder = os.path.join(OUTPUT_PATH, new_name)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # Copy the images
            src_gesture_path = os.path.join(subject_path, gesture_folder)
            for image_name in os.listdir(src_gesture_path):
                src_image = os.path.join(src_gesture_path, image_name)
                
                # We rename the file to include the subject ID to prevent duplicate names
                # e.g., 'frame_01.png' becomes 'subject00_frame_01.png'
                new_image_name = f"subject{subject_folder}_{image_name}"
                dst_image = os.path.join(target_folder, new_image_name)
                
                shutil.copy(src_image, dst_image)

    print("\nSuccess! Data organized into 'data/' folder.")
    print(f"You can now run Step 2.")

if __name__ == "__main__":
    organize_dataset()