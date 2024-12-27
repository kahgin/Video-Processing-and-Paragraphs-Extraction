import cv2
from matplotlib import pyplot as pt

class FindParagraph:
    def __init__(self, image, white_threshold=250, gap_threshold=25):
        '''Initialize the FindParagraph class'''
        self.image = image
        self.binary_image = None
        self.nrow, self.ncol, _ = self.image.shape
        self.rows = []  
        self.paragraph_start_end = []
        self.white_threshold = white_threshold
        self.gap_threshold = gap_threshold

    def binarize_image(self):
        '''Convert image to binary'''
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.binary_image = cv2.threshold(gray_image, self.white_threshold, 255, cv2.THRESH_BINARY)
        return self.binary_image

    def find_rows(self):
        '''Identify rows containing text by checking pixel intensity'''
        self.rows = [row for row in range(self.nrow) if any(pixel_value < self.white_threshold for pixel_value in self.binary_image[row, :])]

    def find_paragraphs(self):
        '''Identify start and end points of paragraphs based on row gaps'''
        start = None
        for i in range(1, len(self.rows)):
            # Check if the rows are consecutive (no gap)
            if self.rows[i] - self.rows[i-1] != 1:
                # If the gap is larger than the threshold, consider it as the end of the paragraph
                if start is not None and self.rows[i] - self.rows[i - 1] > self.gap_threshold:
                    end = self.rows[i - 1]                          # Last non-white row before the gap
                    self.paragraph_start_end.append((start, end))   # Store paragraph range
                    start = self.rows[i]                            # Start a new paragraph after the gap
    
            # If there is no gap, continue the current paragraph
            else:
                if start is None:
                    start = self.rows[i-1]
        
        # Handle the last paragraph, from the last non-white row to the end of the image
        if start is not None:
            last_non_white_row = self.rows[-1]
            end = last_non_white_row
            while end < self.nrow and all(pixel_value >= self.white_threshold for pixel_value in self.binary_image[end, :]):
                end += 1
            
            self.paragraph_start_end.append((start, end))

    def display_image_with_markings(self):
        '''Display the image with markings for paragraph boundaries'''
        pt.figure(figsize=(15, 15))
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pt.imshow(rgb_image)

        # Mark the paragraph boundaries (start and end rows)
        for start, end in self.paragraph_start_end:
            pt.axhline(y=start, color='red', linewidth=1)   # start
            pt.axhline(y=end, color='blue', linewidth=1)    # end

        pt.title("Image with Paragraph Boundaries")
        pt.axis("off")
        pt.show()

    def plot_paragraphs(self):
        '''Plot all paragraphs using Matplotlib'''
        pt.figure(figsize=(15, 15))
        for i, (start, end) in enumerate(self.paragraph_start_end):
            paragraph = cv2.cvtColor(self.image[start:end, :], cv2.COLOR_BGR2RGB)
            pt.subplot(1, len(self.paragraph_start_end), i + 1)
            pt.imshow(paragraph)
            pt.axis('off')
            pt.title(f"Paragraph {i + 1}")

        pt.tight_layout()
        pt.show()
        
    def process_image(self):
        '''Execute the full process to finding paragraphs'''
        self.binarize_image()
        self.find_rows()
        self.find_paragraphs()
        return self.paragraph_start_end

# Example usage
# image_path = "./Columns/002_C1.png"
# extractor = FindParagraph(cv2.imread(image_path))
# extractor.process_image()
# extractor.display_image_with_markings()
# extractor.plot_paragraphs()
