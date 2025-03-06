import rasterio
import numpy as np

def average_pixel_value_total(asc_file, bp_file):
    """
    Computes the average value of all valid pixels in an ASCII Grid (ASC) file.
    It uses the raster's nodata value to ignore invalid pixels. It returns the average value of all valid pixels using the formula:
    average = sim(pixel values)/nsims * nsims/#number of times the pixel was burned (1/Burn probability).

    Inputs:
    - asc_file: Path to the ASCII grid file (float) containing per pixel averages.
    - bp_file: Path to the burn probability ASCII grid file (float).

    Output:
    - average: Average value of all valid pixels in the ASCII grid file (float).

    """
    with rasterio.open(asc_file) as src:
        # Read the first band as a masked array so that nodata values are ignored.
        data = src.read(1, masked=True)

    with rasterio.open(bp_file) as bp:
        burn_probability = bp.read(1, masked = True)
        # Calculate the mean of the valid (non-masked) values.
    pixel_value = data / burn_probability
    average = np.ma.mean(pixel_value)
    return average

# Example usage:
asc_file = "/home/ramiro/Emisiones/50sim/cicatriz_total.asc"  # Replace with your actual ASC file path
bp_file = "/home/ramiro/Emisiones/50sim/bp.asc"
avg_value_total = average_pixel_value_total(asc_file, bp_file)
print("Average pixel value:", avg_value_total)