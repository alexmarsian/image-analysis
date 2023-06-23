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
from kneed import KneeLocator

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
    # Get the ROI and background coordinates for all channels
    coordinates_files = Path(coordinates_path).glob('*.csv')
    channel_coords = {i: {'roi': None, 'bg': None} for i in range(1, tiff_file.shape[2]+1)}
    for c in coordinates_files:
        channel = c.split('_')[2]
        if 'ROI' in c:
            channel_coords[channel]['roi'] = extract_coordinates(c)
        elif 'Background' in c:
            channel_coords[channel]['bg'] = extract_coordinates(c)
    # Find the first frame across all channels after the background has steadied
    first_frame = find_global_first_frame(tiff_file, channel_coords)
    # Stitch the regions of interest into a smaller image
    tiff_file = stitch_regions(tiff_file[first_frame, :, :, :], channel_coords)
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

def extract_coordinates(coordinates_file : str):
    """ 
    Extract the coordinates from the background and ROI position files

    Parameters
    ----------
    coordinates_file : str
        The path to the coordinates file

    Returns
    -------
    pos : list
        A list of lists of ints representing the background coordinates
    r_pos : list
        A list of lists of ints representing the ROI coordinates
    """
    with open(coordinates_file, 'r') as f:
        c = f.readlines()[2:]
        # convert the list of lists to a list of lists of ints
        c = [[int(val) for val in region.strip().split(',')] for region in c]
    return c

def find_global_first_frame(tiff_file: np.ndarray, coordinates: list) -> int:
    """Finds the first frame after the background has steadied across all channels.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced.
        coordinates (dict): The coordinates of the regions of interest, supplied as 
        a dictionary of {channel_number: {roi: coords, bg: coords}}.

    Returns:
        int: The first frame after the background has steadied.
    """
    # extract all the average background intensities for all frames in each channel
    avg_bg_intensities = {}
    start_frames = {}
    for channel in coordinates.keys():
        for region in range(len(coordinates[channel]['bg'])):
            y, x = np.unravel_index(coordinates[channel]['bg'][region], tiff_file.shape[2:])
            # get the average of the region coordinates for all frames
            avg_bg_intensities[channel].append(np.mean(tiff_file[:, channel, y, x], axis=tiff_file.shape[2:]))
        # Now get all the start frames for each channel
        start_frames[channel] = pick_start_frames(avg_bg_intensities[channel])
    # Find the minimum start frame across all channels
    min_start_frame = find_smallest_value(start_frames)
    return int(min_start_frame)

def find_smallest_value(dictionary):
    """Finds the smallest value in an arbitrarily nested dictionary.

    Args:
        dictionary (dict): The dictionary to be searched.

    Returns:
        float: The smallest value in the dictionary.
    """
    smallest = float('inf')  # Initialize with positive infinity
    for value in dictionary.values():
        if isinstance(value, dict):
            # If the value is a nested dictionary, recursively call the function
            nested_smallest = find_smallest_value(value)
            smallest = min(smallest, nested_smallest)
        else:
            # If the value is a number, update the smallest if necessary
            if isinstance(value, (int, float)) and value < smallest:
                smallest = value
    return smallest

# Create a function to pick the start frame for each background based on the knee of the moving average of the background intensity
def pick_start_frames(avg_bg: np.array, ma: int=100, start_threshold: int=1) -> dict:
    """ 
    Pick the start frame for each background based on the knee of the moving average of the background intensity

    Parameters
    ----------
    avg_bg : np.array
        An array of average background intensities for each ROI across all frames
    ma : int
        The number of frames to use to calculate the moving average
    start_threshold : int
        The threshold for the change in background intensity to determine the start frame

    Returns
    -------
    start_frames : dict
        A dictionary of {background_number (int): start_frame (int)}
    """
    start_frames = {}
    for i in range(len(avg_bg)):
        # calculate the moving average across 50 frames
        y = np.convolve(avg_bg[i][:-1], np.ones(ma), 'valid') / ma
        # find the first frame where the change in background intensity is greater than a threshold
        for k in range(len(y)-1):
            if y[k+1] - y[k] > start_threshold:
                start = k
                break
        # Get the knee of this moving average data
        x = np.arange(len(y))
        try:
            kneedle = KneeLocator(x[start:], y[start:], curve='concave', direction='increasing')
        except:
            kneedle = KneeLocator(x[300:], y[300:], curve='concave', direction='increasing')
        # Get the start frame for the background
        start_frames[i] = kneedle.knee
    return start_frames
    
def stitch_regions(tiff_file: np.ndarray, coordinates: list) -> np.ndarray:
    """Stitches the regions of interest into a smaller image.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced.
        coordinates (dict): The coordinates of the regions of interest, supplied as 
        a dictionary of {channel_number: {roi: coords, bg: coords}}.

    Returns:
        np.ndarray: The reduced tiff image.
    """
    # get the set of x and y coordinates for all ROIs for each channel
    x_coords = {}
    y_coords = {}
    for channel in coordinates.keys():
        x_coords[channel] = []
        y_coords[channel] = []
        for region in range(len(coordinates[channel]['roi'])):
            y, x = np.unravel_index(coordinates[channel]['roi'][region], tiff_file.shape[2:])
            # adjust the x and y coords to be the smallest square around the ROI
            y, x = adjust_coordinates(y, x, size=10)
            x_coords[channel] += x
            y_coords[channel] += y
    # extract the regions of interest from the tiff image for each channel
    channel_tiffs = []
    for channel in coordinates.keys():
        channel_tiffs.append(tiff_file[:, channel, y_coords[channel], x_coords[channel]])
    # restack the channels
    tiff_file = np.stack(channel_tiffs, axis=1)
    return tiff_file

def adjust_coordinates(y: int, x: int, size: int=10) -> tuple:
    """Cuts a square of sizexsize around the center of the ROI.
    Args:
        y (int): The y coordinate of the ROI.
        x (int): The x coordinate of the ROI.

    Returns:
        tuple: The adjusted x and y coordinates.
    """
    # Make sure the size is even
    if size % 2 != 0:
        size += 1
    # Find the center of the ROI
    center = (int((max(y) - min(y))/2), int((max(x) - min(x))/2))
    # Generate all possible x,y pairs that form a size*size square around the center
    x = []
    y = []
    for i in range(center[0]-size/2, center[0]+size/2):
        for j in range(center[1]-size/2, center[1]+size/2):
            x.append(j)
            y.append(i)
    return (y, x)


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