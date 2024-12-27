import os
import cv2
import shutil
from column_finder import FindColumn
from paragraph_finder import FindParagraph
from paragraph_validator import ValidateParagraph

class Driver:
    '''Initialize the Driver class'''
    def __init__(self, image_path):
        self.image_path = image_path
        self.column_folder = "Columns"
        self.paragraph_folder = "Paragraphs"
        self.not_paragraph_folder = "Not Paragraphs"
        
        # Run driver
        self.ensure_folders_exist()
        self.extract_paragraphs()
        self.validate_paragraphs()

    def ensure_folders_exist(self):
        '''Create output folders if they do not exist'''
        os.makedirs(self.column_folder, exist_ok=True)
        os.makedirs(self.paragraph_folder, exist_ok=True)
        os.makedirs(self.not_paragraph_folder, exist_ok=True)
        
    def save_image(self, image, folder, name):
        '''Save an image to a folder with a given name'''
        path = os.path.join(folder, name)
        cv2.imwrite(path, image)

    def extract_paragraphs(self):
        '''Extract paragraphs from the columns in the image and save them to the Paragraphs folder'''
        # Step 1: Extract columns
        column_extractor = FindColumn(self.image_path)
        column_extractor.process_image()
        for i, (start, end) in enumerate(column_extractor.column_start_end, start=1):
            column = column_extractor.image[:, start:end]
            column_name = f"{os.path.basename(self.image_path).split('.')[0]}_C{i}.png"
            self.save_image(column, self.column_folder, column_name)

            # Step 2: Extract paragraphs
            paragraph_extractor = FindParagraph(column)
            paragraph_extractor.process_image()
            paragraph_count = 0
            for row_start, row_end in paragraph_extractor.paragraph_start_end:
                paragraph = column[row_start:row_end, :].copy()
                paragraph_count += 1
                paragraph_name = f"{os.path.basename(self.image_path).split('.')[0]}_C{i}_P{paragraph_count}.png"
                self.save_image(paragraph, self.paragraph_folder, paragraph_name)

    def validate_paragraphs(self):
        '''Validate each paragraph in the Paragraphs folder and move invalid ones to the Not Paragraphs folder'''
        for file_name in os.listdir(self.paragraph_folder):
            file_path = os.path.join(self.paragraph_folder, file_name)
            image = cv2.imread(file_path)
            validator = ValidateParagraph(image)
            if not validator.is_paragraph():
                new_path = os.path.join(self.not_paragraph_folder, file_name)
                shutil.move(file_path, new_path)
         