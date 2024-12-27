# Video Processing & Paragraphs Extractor
This program is designed to process videos and extract paragraphs from images of scientific papers. 

Task A: Video Processing

1. Detect whether a video is taken during daytime or nighttime. If nighttime, increase the brightness of the video.
2. Blur all the faces (camera facing) that appear in a video.
3. Resize and overlay a "talking YouTuber" video on the top left of each frame.
4. Add watermarks to protect the videos from being stolen.

Task B: Paragraph Extraction

1. Detect columns from the input images.
2. Extract paragraphs from the detected columns.
3. Validate the extracted paragraphs and save invalid ones in a separate folder.


## Task A Folder Structure & Files
### Folder
- `Input`: Folder containing input video files.
- `Output`: Folder for output video files.
### Files
- `main.py`: Main script to process the video.
- `face_detector.xml`: Haar cascade file for face detection.

## Task B Folder Structure & Files
### Folder
- `Converted Paper`: Folder containing input image files.
- `Columns`: Folder for images of detected columns during processing.
- `Paragraphs`: Folder for final extracted paragraphs.
- `Not Paragraphs`: Folder for non-paragraph content (e.g., images or non-relevant content).
### Files
- `main.py`: Main script.
- `driver.py`: Manages column detection, paragraph extraction, and validation.
- `column_finder.py`: Detect columns from the input images.
- `paragraph_finder.py`: Locate paragraphs within each column of the images.
- `paragraph_validator.py`: Validate paragraphs in an image.

## Prerequisites for Setup
- Python 3.7 or higher
- Spyder IDE (recommended)
- OpenCV 

## Contributors
[Chan Kah Gin](https://github.com/kahgin) ✨ [Ho Zi Shan](https://github.com/Zs1028) ✨ [Lee Wen Xuan](https://github.com/agneslee40) ✨ [Thit Sar Zin](https://github.com/thitsarzin)
