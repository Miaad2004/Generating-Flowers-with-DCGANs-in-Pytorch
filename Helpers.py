import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import pathlib
import random
from tqdm.auto import tqdm
import itertools

class Image_Helper:
    @staticmethod
    def load_image(path):
        return cv2.imread(str(path))
    
class Data_Helper:
    @staticmethod
    def randomly_load_from_directory(directory, n_images_to_load, file_extension='jpg'):
        directory = pathlib.Path(directory)
        image_paths = map(str, list(directory.glob(f"*/*.{file_extension}")))
        random.shuffle(image_paths)
        
        images = [Image_Helper.load_image((image_path)) for image_path in image_paths[0: n_images_to_load]]
        return images
    
    @staticmethod
    def gen_metadata_from_directory(data_path, file_extension='jpg'):
        data_path = pathlib.Path(data_path)
        n_classes = len([path for path in data_path.glob("*") if path.is_dir()])
        
        image_paths = []
        labels = []
        
        for class_dir in tqdm(data_path.iterdir(), total=n_classes):
            # Check if it's a valid directory
            if not class_dir.is_dir():
                pass

            # Add the images to the dataframe
            for image_path in class_dir.glob(f"*.{file_extension}"):
                image_paths.append(image_path)
                labels.append(class_dir.name)
        
        # Create the dataframe                            
        data_frame =  pd.DataFrame({"image_paths": image_paths, "labels": labels})
        
        # Shuffle the dataframe
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        
        return data_frame, n_classes
        
    @staticmethod
    def plot(images, titles=None, figSize=(10, 10), isBGR=True, nCols=7):
        """
        This method allows you to display a collection of images in a grid, with titles for each image.

        Parameters:
        - `images` (list): A list of images.
        - `titles` (list, optional): A list of strings, representing the title for each image. The length of `titles` should be equal to the length of `images`.
        - `figSize` (tuple): The size of the figure.
        - `isBGR` (bool, optional): Indicates whether the input images are in BGR format (default is True).
        - `nCols` (int, optional): The number of columns in the grid (default is 3).

        Returns:
        None
        """

        # Check if the images and the titles have the same length
        if titles is not None and len(images) != len(titles):
            raise ValueError("Images and titles must have the same length")

        # Calculate the number of rows required for the figure
        nRows = int(np.ceil(len(images) / nCols))

        # Create the fig
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=figSize)

        for ax, img, title in itertools.zip_longest(axes.flat, images, titles or []):
            if isBGR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img)
            if titles is not None:
                ax.set_title(title)

            # Remove the tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        fig.tight_layout()
