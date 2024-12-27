import cv2
import numpy as np
import os


## Q1
def calculate_brightness(frame):
    '''Adjust Brightness for nighttime video if detected'''

    # Convert current frame to grayscale (colour information is not necessary to calculate brightness)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate average brightness of the frame by calculating the mean of all pixel values
    brightness = np.mean(gray_frame)
    return brightness


def increase_brightness(frame, factor=2.0):
    '''Increase the brightness of the frame by multiplying all pixel values with the factor value'''

    bright_frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)
    return bright_frame


def adjust_brightness_for_video(video_file):
    '''Detect whether the video is taken during the day or night and increase the brightness of nighttime videos'''

    print(f"\nProcessing Video for Brightness Adjustment: {video_file}")
    
    vid = cv2.VideoCapture(video_file)                          # Read the video file
    total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))    # Obtain total number of frames in the video
    brightness_values = []                                      # A list to store the brightness of each frame

    # Loop through each frame in the video to calculate its brightness
    for frame_count in range(total_no_frames):
        success, frame = vid.read()
        if not success:
            break
        brightness = calculate_brightness(frame) # Calculate brightness for the current frame
        brightness_values.append(brightness)
    
    # Calculate the average brightness of the entire video
    avg_brightness = np.mean(brightness_values)
    print(f"Average Brightness: {avg_brightness:.2f}")
    
    # If the average brightness is less than 100, then it is considered as nighttime
    is_nighttime = avg_brightness < 100
    day_night_status = "NIGHT" if is_nighttime else "DAY"
    print(f"Detected as: {day_night_status}")
    
    # Loop through the frames again and adjust brightness if it's nighttime
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)             # Reset to the first frame
    frames = []                                     # List to store processed frames
    for frame_count in range(total_no_frames):
        success, frame = vid.read()
        if not success:
            break
        if is_nighttime:
            frame = increase_brightness(frame, 2.0) # Increase the brightness by a factor of 2
        frames.append(frame)

    vid.release()
    return frames


## Q2 & Q3
def blur_faces_in_video(main_video_frames, talking_video_path, face_cascade_path):
    '''Blur faces in the video + Add overlay video at top left corner'''

    print("Processing Video for Face Blurring and video overlay... ")
    
    # Open the talking video which will be overlaid in the top left corner of the main video
    talking_video = cv2.VideoCapture(talking_video_path)

    frame_width = main_video_frames[0].shape[1]     # Obtain frames' width in the main video
    frame_height = main_video_frames[0].shape[0]    # Obtain frames' height in the main video
    total_frames = len(main_video_frames)           # Obtain total number of frames in the main video

    # Load pre-trained face detection model (Haar Cascade Classifier)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Resize the talking video frame to 30% of the main video size for the overlay
    resize_width = int(frame_width * 0.3)
    resize_height = int(frame_height * 0.3)

    
    processed_frames = [] # List to store processed frames with blurred faces and overlay
    
    # Process each frame in the main video
    for frame_idx in range(total_frames):
        main_frame = main_video_frames[frame_idx]
        gray_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale for face detection
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5) # Detect faces

        # Blur detected faces
        for (x, y, w, h) in faces:
            face_roi = main_frame[y:y + h, x:x + w] # Region of Interest (ROI) where the face is
            blurred_face = cv2.GaussianBlur(face_roi, (35, 35), 0) # Apply Gaussian Blur to the face
            main_frame[y:y + h, x:x + w] = blurred_face # Replace the original face with the blurred one

        # Overlay talking video on the top left corner
        success_talking, talking_frame = talking_video.read()
        if not success_talking:
            talking_video.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop talking video continuously once ended
            success_talking, talking_frame = talking_video.read()

        # Resize the talking frame and place it in the top left corner of the main frame
        talking_frame_resized = cv2.resize(talking_frame, (resize_width, resize_height)) # Resize overlay
        x_offset, y_offset = 10, 10 # Place overlay at top left corner
        main_frame[y_offset:y_offset + resize_height, x_offset:x_offset + resize_width] = talking_frame_resized

        processed_frames.append(main_frame)

    talking_video.release()
    return processed_frames


## Q4 & Q5
def add_watermarks_and_end_screen(frames, watermark1_path, watermark2_path, end_screen_path):
    '''Add watermark + Add end screen'''

    print("Processing Video for Watermarking and End Screen... ")

    # Load watermark images and the end screen video
    watermark1 = cv2.imread(watermark1_path)
    watermark2 = cv2.imread(watermark2_path)
    end_screen_video = cv2.VideoCapture(end_screen_path)

    fps = 30 # Set frames per second to 30 

    frames_per_watermark = fps * 5  # Change watermark every 5 seconds
    frame_count = 0

    processed_frames = [] # Store frames with watermarks
    
    # Apply watermark to every frame in the video
    for frame in frames:
        # Switch between two watermarks every 5 seconds
        current_watermark = watermark1 if (frame_count // frames_per_watermark) % 2 == 0 else watermark2
        blended_frame = cv2.addWeighted(frame, 1.0, current_watermark, 1.0, 0) # Blend the watermark with the frame
        processed_frames.append(blended_frame)
        frame_count += 1

    # Add end screen at the end of the video
    while True:
        ret, frame = end_screen_video.read()
        if not ret:
            break
        processed_frames.append(frame)

    end_screen_video.release()
    return processed_frames


## Execute the full processing pipeline
def process_videos(videos, input_folder=None, output_folder=None):
    '''Function to pass videos for processing'''
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the list of videos to process
    for video in videos:
        file_path = os.path.join(input_folder, video) if input_folder else video
        print(f"Processing video: {video}")

        # Step 1: Adjust brightness based on whether it's daytime or nighttime
        brightness_frames = adjust_brightness_for_video(file_path)

        # Step 2: Blur faces in video + add overlay of talking video at top left
        face_blurred_frames = blur_faces_in_video(brightness_frames, os.path.join(input_folder,"talking.mp4"), "face_detector.xml")

        # Step 3: Add watermark + Add end screen
        final_frames = add_watermarks_and_end_screen(
            face_blurred_frames, 
            os.path.join(input_folder,"watermark1.png"), 
            os.path.join(input_folder,"watermark2.png"), 
            os.path.join(input_folder,"endscreen.mp4"))

        # Step 4: Save final processed video
        output_video_path = os.path.join(output_folder, f"final_{video.split('.')[0]}.avi")
        frame_height, frame_width = final_frames[0].shape[:2] # Get frame size from processed frames
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Set the video codec
        output_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

        for frame in final_frames:
            output_video.write(frame) # Write each frame to the output video

        output_video.release() # Save video
        print(f"Processed video saved as: {output_video_path}\n")

    print("Processing complete.")

# Input video folder
input_folder = "Input"

# Output video folder
output_folder = "Output"

# List of video files
videos = ["alley.mp4", "office.mp4", "singapore.mp4", "traffic.mp4"]

# Start processing video
process_videos(videos, input_folder, output_folder)
