import numpy as np
import cv2
from matplotlib import pyplot as pt

class FindColumn:
    def __init__(self, image_path, white_threshold=250, min_col_width=10):
        '''Initialize the FindColumn class'''
        self.image_path = image_path
        self.image = cv2.imread(self.image_path, 3)
        self.binary_image = None
        self.nrow, self.ncol, _ = self.image.shape
        self.white_col = []
        self.column_start_end = []
        self.white_threshold = white_threshold
        self.min_col_width = min_col_width

    def binarize_image(self):
        '''Convert image to binary'''
        _, self.binary_image = cv2.threshold(self.image, self.white_threshold, 255, cv2.THRESH_BINARY)
        return self.binary_image

    def find_white_col(self, hist_proj=False, threshold=0.95):
        '''Identify white columns based on vertical projections'''
        vertical_projection = np.sum(self.binary_image, axis=0)
        peak_threshold = threshold * np.max(vertical_projection)
        self.white_col = np.where(vertical_projection > peak_threshold)[0]
        
        # Plot the histogram projection if required
        if hist_proj:
            self.plot_hist_proj(vertical_projection, peak_threshold)
        
    def plot_hist_proj(self, vertical_projection, peak_threshold):
        '''Plot histogram projection'''
        pt.figure(figsize=(10, 5))
        pt.plot(vertical_projection)
        pt.title("Vertical Histogram Projection")
        pt.xlabel("Column Index")
        pt.ylabel("Pixel Intensity Sum")
        pt.axhline(peak_threshold, color='red', linestyle='--', label="Threshold")
        pt.legend()
        pt.show()

    def find_col(self):
        '''Identify start and end of each column & filter valid regions'''
        start = None
        for i in range(1, len(self.white_col)):
            if self.white_col[i] - self.white_col[i - 1] != 1: # Detect column boundaries.
                if start is None:
                    start = self.white_col[i - 1]
                if (self.white_col[i] - start) > self.min_col_width: # Filters valid regions.
                    self.column_start_end.append((start, self.white_col[i]))
                start = None  # Reset start
    
    def display_image_with_markings(self):
        '''Display the image with markings for column boundaries'''
        pt.figure(figsize=(10, 10))
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pt.imshow(rgb_image)

        # Mark the column boundaries (start and end column)
        for start, end in self.column_start_end:
            pt.axvline(x=start, color='red', linewidth=1)   # start
            pt.axvline(x=end, color='blue', linewidth=1)    # end

        pt.title("Image with Column Boundaries")
        pt.axis("off")
        pt.show()

    
    def plot_columns(self):
        '''Plot all columns using Matplotlib'''
        pt.figure(figsize=(10, 10))
        for i, (start, end) in enumerate(self.column_start_end):
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            column = image[:, start:end]
            pt.subplot(1, len(self.column_start_end), i + 1)
            pt.imshow(column, cmap='gray')
            pt.axis('off')
            pt.title(f"Column {i + 1}")
        pt.tight_layout()
        pt.show()

    def process_image(self):
        '''Execute the full process for finding columns'''
        self.binarize_image()
        self.find_white_col(hist_proj=False)
        self.find_col()
        return self.column_start_end

# Example usage
# image_path = "./Converted Paper/006.png"
# extractor = FindColumn(image_path)
# extractor.process_image()
# extractor.display_image_with_markings()
# extractor.plot_columns()
