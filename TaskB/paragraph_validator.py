import cv2
import numpy as np

class ValidateParagraph:
    def __init__(self, image):
        '''Initialize the ValidateParagraph class'''
        self.image = image
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def is_paragraph(self, white_threshold=250, line_gap_threshold=1, row_threshold=3, col_threshold=5):
        '''Verify if the input image meets paragraph requirements'''
        # Binarize the image
        _, binary_image = cv2.threshold(self.gray_image, white_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Compute row and column densities
        row_density = np.sum(binary_image, axis=1) > 0
        col_density = np.sum(binary_image, axis=0) > 0

        # Count non-empty rows and columns
        num_rows = np.sum(row_density)
        num_cols = np.sum(col_density)
    
        # Check for intermediate white lines
        white_line_indices = np.where(~row_density)[0]  # Rows with no text
        line_gaps = np.diff(white_line_indices)  # Gaps between white lines
        has_intermediate_white_lines = np.any(line_gaps >= line_gap_threshold)
    
        # Final conditions: must meet row and column thresholds and have intermediate white lines
        return num_rows >= row_threshold and num_cols >= col_threshold and has_intermediate_white_lines
