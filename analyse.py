# This file contains the main function for the image analysis program.
# It takes in a path to the tiff images and their metadata file,
# extracts only the regions of interest from the first frame after background has steadied,
# stitches these regions into a smaller, single tiff image,
# correcting the frame order based on the metadata file,
# and then feeds this smaller image into a Python implementation of the Picasso algorithm.

from PIL import Image
import numpy as np
from pathlib import Path
import tifffile as tiff
import click
from typing import Union

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('coordinates_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=False))
@click.option('--metadata', type=Union[str, None], default='metadata.txt', help='Path to metadata file')
def main(input_path: str, coordinates_path: str, output_path: str, metadata: Union[str, None]):
    """A program to analyse tiff images of cells.

    Args:
        input_path (str): Path to the tiff images. 
        coordinates_path (str): Path to the coordinates file.
        output_path (str): Path to the output file.
        metadata (Union[str, None]): Path to the metadata file, if it exists.
    """
    # Load the tiff image
    t = tiff.imread(input_path)
    # Reduce the tiff image
    t = reduce_tiff(t, coordinates_path, metadata)
    # Save the reduced tiff image
    tiff.imwrite(output_path, t)
    # Pass the reduced tiff image into picasso
    run_picasso(output_path)

def reduce_tiff(tiff_file: np.ndarray, coordinates_path: str, metadata: Union[str, None]) -> np.ndarray:
    """Reduces the size of the tiff image by extracting only the regions of interest.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced.
        coordinates_path (str): Path to the coordinates file.
        metadata (Union[str, None]): Path to the metadata file, if it exists.

    Returns:
        np.ndarray: The reduced tiff image. With frame order corrected if metadata is provided.
    """
    # Get the ROI and background coordinates
    coordinates = extract_coordinates(tiff_file, coordinates_path)
    # Find the first frame after the background has steadied
    first_frame = find_first_frame(tiff_file, coordinates)
    # Stitch the regions of interest into a smaller image
    tiff_file = stitch_regions(tiff_file[first_frame, :, :, :], coordinates)
    # Correct the frame order if metadata is provided
    if metadata:
        tiff_file = correct_frame_order(tiff_file, metadata)
    return tiff_file

def run_picasso(tiff_path: str) -> None:
    """Runs the picasso algorithm on the reduced tiff image.

    Args:
        tiff_path (str): Path to the reduced tiff image.
    """
    #TODO: Implement this function

# First extract the coordinates for both channels
def extract_coordinates(background_pos_file : str, roi_pos_file :str):
    """ 
    Extract the coordinates from the background and ROI position files

    Parameters
    ----------
    background_pos_file : str
        The path to the background position file
    roi_pos_file : str
        The path to the ROI position file

    Returns
    -------
    bg_pos : list
        A list of lists of ints representing the background coordinates
    r_pos : list
        A list of lists of ints representing the ROI coordinates
    """
    with open(background_pos_file, 'r') as f:
        bg_pos = f.readlines()[2:]
        # convert the list of lists to a list of lists of ints
        bg_pos = [[int(val) for val in region.strip().split(',')] for region in bg_pos]
    with open(roi_pos_file, 'r') as f:
        r_pos = f.readlines()[2:]
        # convert the list of lists to a list of lists of ints
        r_pos = [[int(val) for val in region.strip().split(',')] for region in r_pos]
    return bg_pos, r_pos

def find_first_frame(tiff_file: np.ndarray, coordinates: list) -> np.ndarray:
    """Finds the first frame after the background has steadied.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced.
        coordinates (list): The coordinates of the regions of interest.

    Returns:
        np.ndarray: The first frame after the background has steadied.
    """
    #TODO: Implement this function

def stitch_regions(tiff_file: np.ndarray, coordinates: list) -> np.ndarray:
    """Stitches the regions of interest into a smaller image.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced.
        coordinates (list): The coordinates of the regions of interest.

    Returns:
        np.ndarray: The reduced tiff image.
    """
    #TODO: Implement this function

def correct_frame_order(tiff_file: np.ndarray, metadata: str) -> np.ndarray:
    """Corrects the frame order of the tiff image based on the metadata file.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced.
        metadata (str): Path to the metadata file.

    Returns:
        np.ndarray: The reduced tiff image.
    """
    #TODO: Implement this function

if __name__ == "__main__":
    main()