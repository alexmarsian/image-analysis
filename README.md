# image-analysis
Initial code for swan genomics data processing

# Usage
Run the analyse.py script from the command line, or import run from analyse to use in other files.
Command line arguments:
- input_path: the path to the input tiff stack
- coordinates_path: path to the folder containing the coordinates files
- output_path: path to save the new tiff stack with dimensions frames x channels x rois x size x size
- metadata (optional): path to the metadata file
- size (int): the size of the square to be cut around the centre of a region of interest, e.g., 10 = 10x10 square per ROI.

# Progress/To-Do
- Only been tested on the sample data provided by James which contains only two channels, also coordinate file parsing is based on the file naming system provided in this sample (E.g., Expanded_Channel_1_Background_Positions.csv)
- Gaussian fit is done without any threshold (we talked about doing a gradient threshold but hadn't decided on what it should be)
- Gaussian fit is using scipy (faster than lmfit), but could still be optimised further.

# Example

python analyse.py data/sample/test.tif data/sample/test_data data/sample/test_output.tif --size 10
