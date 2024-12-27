import os
from driver import Driver

# Define input folder and file names
input_folder = "Converted Paper"
file_names = ["001.png", "002.png", "003.png", "004.png", "005.png", "006.png", "007.png", "008.png"]

# Process each image
for file_name in file_names:
    file_path = os.path.join(input_folder, file_name)
    print(f"Processing file: {file_name}")

    # Run the driver
    driver = Driver(image_path=file_path)

print("Processing complete.")
