import cv2
import face_recognition
import os
import numpy as np
import logging
import pandas as pd
import time
from datetime import datetime
import pickle  # Corrected typo
import hashlib
import argparse
import pyttsx3
from ultralytics import solutions

# ----------------------------
# Configuring Logging and Attendance File
# -----------------------------

# Configure logging to display INFO and above levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the current date and format it as dd-mm-yyyy
current_date = datetime.now().strftime("%d-%m-%Y")
# File to store attendance records in Excel format
attendance_file = f'attendance_{current_date}.xlsx'

# --------------------------------
# 2. Loading Known Faces from Subdirectories with Caching
# --------------------------------

CACHE_DIR = "cache"
ENCODINGS_FILE = os.path.join(CACHE_DIR, f"encodings_{current_date}.pkl")
HASH_FILE = os.path.join(CACHE_DIR, f"encodings_hash_{current_date}.txt")

def compute_dir_hash(folder_path):
    '''Computes a hash of the folder's contents to detect changes.'''
    hash_md5 = hashlib.md5()
    for root, dirs, files in os.walk(folder_path):
        for fname in sorted(files):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".webp")):
                file_path = os.path.join(root, fname)  # Corrected
                try:
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                except Exception as e:
                    logging.error(f"Failed to read file '{file_path}' for hashing: {e}")
    return hash_md5.hexdigest()  # Indentation fixed


def load_known_faces_with_cache(folder_path, cache_file=ENCODINGS_FILE, hash_file=HASH_FILE):
    """Load known face encodings and names, utilizing cache with automatic invalidation."""
    known_encodings = []
    known_names = []

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    current_hash = compute_dir_hash(folder_path)

    if os.path.isfile(cache_file) and os.path.isfile(hash_file):  # Corrected
        try:
            with open(hash_file, "r") as f:
                cached_hash = f.read()  # Corrected
            if current_hash == cached_hash:
                # Load from cache
                with open(cache_file, "rb") as f:  # Corrected
                    known_encodings, known_names = pickle.load(f)
                logging.info(f"Loaded cached encodings from '{cache_file}'.")
                return known_encodings, known_names
            else:
                logging.info("Changes detected in known faces directory. Reprocessing images.")
        except Exception as e:
            logging.error(f"Failed to load or compare hash: {e}")

    # If cache is invalid or does not exist, process images
    known_encodings, known_names = _process_images_and_encode(folder_path)

    # Save encodings and hash to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((known_encodings, known_names), f)
        with open(hash_file, 'w') as f:
            f.write(current_hash)  # Corrected
        logging.info(f'Cache encodings saved to {cache_file} with updated hash.')
    except Exception as e:
        logging.error(f'Failed to save cache or hash: {e}')

    return known_encodings, known_names

def _process_images_and_encode(folder_path):
    '''Helper function to process images and encode faces.'''
    known_encodings = []
    known_names = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    if not os.path.exists(folder_path):
        logging.error(f'The folder {folder_path} does not exist. Please create it and add known faces.')
        return known_encodings, known_names

    # Iterate through each subdirectory in the known_faces folder
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):  # Corrected
            logging.warning(f'{person_folder} is not a directory. Skipping.')
            continue

        for filename in os.listdir(person_folder):  # Corrected
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(person_folder, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        logging.warning(f'Unable to read image {img_path}. Skipping.')
                        continue
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Corrected
                    encodings = face_recognition.face_encodings(rgb_img)
                    if encodings:
                        for encoding in encodings:
                            known_encodings.append(encoding)
                            known_names.append(person_name)
                        logging.info(f'Loaded face encoding for {person_name} from {filename}.')
                    else:
                        logging.warning(f'No faces found in image {img_path}.')
                except Exception as e:
                    logging.error(f'Error processing {img_path}: {e}')

    logging.info(f'Total known faces loaded: {len(known_encodings)}')
    return known_encodings, known_names

def clear_cache(cache_file=ENCODINGS_FILE, hash_file=HASH_FILE):
    '''Clears the cached encodings by deleting the cache and hash files.'''
    if os.path.isfile(cache_file):
        try:
            os.remove(cache_file)
            logging.info(f'Cache file {cache_file} has been deleted.')
        except Exception as e:
            logging.error(f'Failed to delete cache file {cache_file}: {e}')
    else:
        logging.info(f"No cache file found at '{cache_file}'.")

    if os.path.isfile(hash_file):
        try:
            os.remove(hash_file)
            logging.info(f'Hash file {hash_file} has been deleted.')
        except Exception as e:
            logging.error(f'Failed to delete hash file {hash_file}: {e}')
    else:
        logging.info(f'No hash file found at {hash_file}.')



#--------------------------------
# Startup function
#--------------------------------
'''def startup_speech(startup):
    if startup== 1:
        speech1= print('Welcome to JO\'s Enterprise, please move to the reception')
    elif startup==2:
        return welcome_speech(speech)
        '''

#--------------------------------------
# Object Counting
#-------------------------------------
region_points = [(300, 10), (300, 500), (400, 500), (400, 10)]# For Vertical positions can be used for Humans, animals etc.


counter= solutions.ObjectCounter(
    show= True,
    region= region_points,
    model= "yolo11n.pt"
    )
        


#--------------------------------
# Beginning speech before detection
#--------------------------------


def welcome_speech(speech, voice_type='male', save_audio=False):
    
    # Initializing the pyttsx3 engine
    engine= pyttsx3.init()

    # Getting available voices
    voices= engine.getProperty("voices")

    # Set properties: voice, speech, rate and volume (optional)
    engine.setProperty('voice', voices[1].id) # This is for female voice. (2) is for male
    engine.setProperty('rate', 150) # Speed of speech
    engine.setProperty('volume', 1) # Volume (0.0 to 1.0)

    #speak the text
    engine.say(speech)

    # Run the speech engine
    engine.runAndWait()



# -------------------------
# 3. Recording attendance
# ----------------------------

def record_attendance(name):
    '''Record the attendance of a recognized individual by adding their name and timestamp to an excel file.'''
    try:
        # Check if attendance file exists, if not create
        if not os.path.isfile(attendance_file):
            df = pd.DataFrame(columns=['Name', 'Timestamp'])
            df.to_excel(attendance_file, index=False)
            logging.info(f'Created new attendance file: {attendance_file}.')

        # Load existing attendance records
        df = pd.read_excel(attendance_file)  # Corrected

        # Append attendance if the name is not already in the records
        if name not in df['Name'].values:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Corrected
            new_record = pd.DataFrame({'Name': [name], 'Timestamp': [timestamp]})  # Corrected

            # Concatenate the existing DataFrame with the new record
            df = pd.concat([df, new_record], ignore_index=True)

            # Write the updated DataFrame back to the excel file
            df.to_excel(attendance_file, index=False)

            log_info= logging.info(f'Attendance recorded for {name} at {timestamp}.')
        else:
            logging.info(f'{name} already recorded in attendance for today.')
    except Exception as e:
        logging.error(f'Failed to record attendance for {name}: {e}')

# ---------------------
# 4. Processing video
# ---------------------

def process_frame(frame, known_encodings, known_names, distance_threshold=0.5):
    '''Detect face in a frame, recognize known faces, draw bounding boxes and labels, and record attendance.'''
    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Corrected

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Corrected

    recognized_names = []  # Keep track of recognized names for this frame

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compute distance between this face and all known faces
        distances = face_recognition.face_distance(known_encodings, face_encoding)  # Corrected
        name = 'Unknown'

        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            if best_distance < distance_threshold:
                name = known_names[best_match_index]
                recognized_names.append(name)  # Track recognized names
                logging.debug(f'Recognized {name} with distance {best_distance:.2f}.')
            else:
                logging.info(f'No match found within threshold for distance {best_distance:.2f}.')

        # Scale back face locations to original frame size
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        
        # Draw label with name below the face
        y_position = bottom + 20 if bottom + 20 < frame.shape[0] else bottom - 10
        cv2.putText(frame, name, (left, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Corrected

    # RECORD ATTENDANCE FOR RECOGNIZED NAMES
    for name in set(recognized_names):  # Use set to avoid duplicate attendance
        record_attendance(name)

    return frame  # Return the annotated frame for display

# ---------------------------------
# 5. Main Function
# ---------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Real-Time Face Recognition Attendance System')
    parser.add_argument('--clear-cache', action='store_true', help='Clear the cached encodings and recognize faces')
    return parser.parse_args()


def main():
    '''Main function to run the face recognition attendance system.'''
    args = parse_arguments()

    known_faces_folder = 'KnownFaces'

    if args.clear_cache:
        clear_cache(ENCODINGS_FILE, HASH_FILE)

    known_encodings, known_names = load_known_faces_with_cache(known_faces_folder)
    if not known_encodings:
        logging.error('No known faces loaded. Please add known face images to the "KnownFaces" folder.')
        return

    # Initialize video capture from IP camera or switch to 0 for default webcam
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        logging.error('Could not open video source')
        return

    try:
        prev_time = time.time()
        while True:
            ret, frame = video.read()
            if not ret:
                logging.error('Failed to grab frame from video source.')
                break

            # Process the frame for face recognition and attendance
            annotated_frame = process_frame(frame, known_encodings, known_names, distance_threshold=0.4)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0  # Corrected
            prev_time = current_time

            # Display FPS on the frame
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Adjust as needed

            # Resize frame for display (optional)
            display_frame = cv2.resize(annotated_frame, (640, 480))  # Adjust as needed

            #Imshow
            cv2.imshow('video', display_frame)
            


            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # 27 is the Esc key
                logging.info('Exiting the attendance system.')
                break
    except KeyboardInterrupt:
        logging.info('Interrupted by user. Exiting...')
    finally:
        # Release video capture and close windows
        video.release()
        cv2.destroyAllWindows()
        logging.info('Resources released and windows closed.')

'''def options(op):
    if op==1:
        sound= print("Welcome to the reception, please move to the reception office")
    elif op==2:
        return main()'''

# startup_speech= 'Welcome to JO\'s Enterprise, Please press "1" for visitor and "2" for employer'
startup_speech= 'Welcome to JO\'s Enterprise, please scan face'
        

if __name__ == "__main__":
    welcome_speech(startup_speech)
    main()
