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
import lmfit

@click.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('coordinates_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(exists=False, path_type=Path))
@click.option('--metadata', type=Union[str, None], default='metadata.txt', help='Path to metadata file')
@click.option('--size', type=int, default=10, help='Size of the square to be cut around the center of the ROI')
def main(input_path: click.Path, coordinates_path: click.Path, output_path: click.Path, metadata: Union[str, None], size: int):
    """A program to analyse tiff images of cells.

    Args:
        input_path (str): Path to the tiff images. 
        coordinates_path (str): Path to the coordinates file.
        output_path (str): Path to the output file.
        metadata (Union[str, None]): Path to the metadata file, if it exists.
        size (int, optional): The size of the square to be cut around the center of the ROI. Defaults to 10.
    """
    run(input_path, coordinates_path, output_path, metadata, size)

# Adding non-click version of main function for import into other scripts
def run(input_path: Union[Path, str], coordinates_path: Union[Path, str], output_path: Union[Path, str], metadata: Union[Path, str, None], size: int=10):
    """A program to analyse tiff images of cells.

    Args:
        input_path (str): Path to the tiff images. 
        coordinates_path (str): Path to the coordinates file.
        output_path (str): Path to the output file.
        metadata (Union[str, None]): Path to the metadata file, if it exists.
        size (int, optional): The size of the square to be cut around the center of the ROI. Defaults to 10.
    """
    # Load the tiff image
    t = tiff.imread(input_path)
    # Reduce the tiff image
    t = reduce_tiff(t, coordinates_path, metadata, size)
    # Save the reduced tiff image
    tiff.imwrite(output_path, t, shape=t.shape)
    # Pass the reduced tiff image into picasso
    fit_gaussians(output_path)

def reduce_tiff(tiff_file: np.ndarray, coordinates_path: Union[Path, str], metadata: Union[str, None], size: int=10) -> np.ndarray:
    """Reduces the size of the tiff image by extracting only the regions of interest.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced.
        coordinates_path (str): Path to the coordinates file.
        metadata (Union[str, None]): Path to the metadata file, if it exists.
        size (int, optional): The size of the square to be cut around the center of the ROI. Defaults to 10.

    Returns:
        np.ndarray: The reduced tiff image. With frame order corrected if metadata is provided.
    """
    # Get the ROI and background coordinates for all channels
    coordinates_files = Path(coordinates_path).glob('*.csv')
    coordinates_files = [c for c in coordinates_files if 'Positions' in str(c)]
    channel_coords = {str(i): {'roi': None, 'bg': None} for i in range(1, tiff_file.shape[1]+1)}
    for c in coordinates_files:
        channel = str(c.stem).split('_')[2]
        if 'ROI' in str(c.stem):
            channel_coords[channel]['roi'] = extract_coordinates(c)
        elif 'Background' in str(c.stem):
            channel_coords[channel]['bg'] = extract_coordinates(c)
    # Find the first frame across all channels after the background has steadied
    first_frame = find_global_first_frame(tiff_file, channel_coords)
    # Stitch the regions of interest into a smaller image
    tiff_file = stitch_regions(tiff_file[first_frame:, :, :, :], channel_coords, size=size)
    # Correct the frame order if metadata is provided
    if metadata:
        tiff_file = correct_frame_order(tiff_file, metadata)
    return tiff_file

def fit_gaussians(tiff_path: str) -> None:
    """Fits 2D guassians to all ROIs across all channels in the tiff image.
    Only fits for regions that have a signal to noise ratio greater than 3.

    Args:
        tiff_path (str): Path to the reduced tiff image with shape (frames, channels, num_rois, size, size).
    """
    print('Fitting gaussians...')
    # Create a 2D gaussian model
    model = lmfit.models.Gaussian2dModel()
    tiff_file = tiff.imread(tiff_path)
    gaussian_params = {}
    count = 0
    frames, channels, num_rois, _,__ = tiff_file.shape
    for channel in range(channels):
        for frame in range(frames):
            for roi in range(num_rois):
                subtiff = tiff_file[frame, channel, roi, :, :]
                # Find x and y coordinates of the maximum value in subframe
                maxi_y, maxi_x = np.unravel_index(subtiff.argmax(), subtiff.shape)
                # Fit 2D gaussian on the subframe using lmfit, centred on the maximum value
                # Create a parameters object fo
                params = model.make_params()
                # Set the initial guesses for the parameters
                params['amplitude'].set(value=subtiff[maxi_y, maxi_x])
                params['centerx'].set(value=maxi_x)
                params['centery'].set(value=maxi_y)
                params['sigmax'].set(value=1)
                params['sigmay'].set(value=1)
                # Fit the model to the data
                result = model.fit(subtiff, params, x=np.arange(subtiff.shape[1]), y=np.arange(subtiff.shape[0]),)
                # Get the parameters of the fit
                gaussian_params[(channel, frame, roi)] = result.params
                count += 1
                if count % 100 == 0:
                    print(count / (num_rois * frames * 2))

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

def find_global_first_frame(tiff_file: np.ndarray, coordinates: dict) -> int:
    """Finds the first frame after the background has steadied across all channels.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced.
        coordinates (dict): The coordinates of the regions of interest, supplied as 
        a dictionary of {channel_number: {roi: coords, bg: coords}}.

    Returns:
        int: The first frame after the background has steadied.
    """
    # extract all the average background intensities for all frames in each channel
    avg_bg_intensities = {i: [] for i in coordinates.keys()}
    start_frames = {}
    for channel in coordinates.keys():
        for region in range(len(coordinates[channel]['bg'])):
            y, x = np.unravel_index(coordinates[channel]['bg'][region], tiff_file.shape[2:])
            # get the average of the region coordinates for all frames
            substack = tiff_file[:, int(channel)-1, y, x]
            avg_bg_intensities[channel].append(np.mean(substack, axis=1))
        # Now get all the start frames for each channel
        start_frames[channel] = pick_start_frames(avg_bg_intensities[channel])
    # Find the minimum start frame across all channels
    return find_smallest_value(start_frames)

def find_smallest_value(dictionary):
    """Finds the smallest value in an arbitrarily nested dictionary.

    Args:
        dictionary (dict): The dictionary to be searched.

    Returns:
        float: The smallest value in the dictionary.
    """
    smallest = float('inf')  # Initialize with positive infinity
    for key in dictionary.keys():
        smallest = min(smallest, min(dictionary[key].values()))
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
    
def stitch_regions(tiff_file: np.ndarray, coordinates: list, size: int=10) -> np.ndarray:
    """Stitches the regions of interest into a smaller image.

    Args:
        tiff_file (np.ndarray): The tiff image to be reduced, of shape (frames, channels, y, x).
        coordinates (dict): The coordinates of the regions of interest, supplied as 
        a dictionary of {channel_number: {roi: coords, bg: coords}}.
        size (int, optional): The size of the square to be cut around the center of the ROI. Defaults to 10.

    Returns:
        np.ndarray: The reduced tiff image with shape (frames, channels, num_rois, size, size).
        where the num_rois is the maximum number of ROIs across all channels, and 
        any channels with fewer ROIs than this are padded with arrays of zeros.
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
            y, x = adjust_coordinates(y, x, size=size)
            x_coords[channel] += x
            y_coords[channel] += y
    # extract the regions of interest from the tiff image for each channel
    channel_tiffs = []
    for channel in coordinates.keys():
        channel_tiffs.append(tiff_file[:, int(channel)-1, y_coords[channel], x_coords[channel]])
    # get the number of rois for each channel
    num_rois = [len(coordinates[channel]['roi']) for channel in coordinates.keys()]
    # reshape the tiff images to be of shape (frames, num_rois, size, size)
    for i in range(len(channel_tiffs)):
        channel_tiffs[i] = np.reshape(channel_tiffs[i], (channel_tiffs[i].shape[0], num_rois[i], size, size))
    # if the number of rois is not the same for each channel, pad the smaller channels with size*size arrays of zeros
    max_rois = max(num_rois)
    for i in range(len(channel_tiffs)):
        if num_rois[i] < max_rois:
            channel_tiffs[i] = np.concatenate((channel_tiffs[i], np.zeros((channel_tiffs[i].shape[0], max_rois-num_rois[i], size, size))), axis=1)
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
    center = (int((max(y) - min(y))//2), int((max(x) - min(x))//2))
    # Generate all possible x,y pairs that form a size*size square around the center
    x = []
    y = []
    for i in range(center[0]-size//2, center[0]+size//2):
        for j in range(center[1]-size//2, center[1]+size//2):
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
    return tiff_file

if __name__ == "__main__":
    main()