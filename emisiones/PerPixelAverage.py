import os
import numpy as np
import rasterio

input_folder = "/home/ramiro/Emisiones/50sim/CFB_manual/"
output_file = "/home/ramiro/Emisiones/50sim/cicatriz_total_copa.asc" 

def average_asc_files(input_folder, output_file):
    """
    Computes the per pixel average of all valid pixel values in a set of ASCII grids. It uses the rasters nodata value to ignore 
    invalid pixels. The average is computed as the sum of al valid pixel values divided by the number of input files.
    
    Input:
    - input_folder: Path to the folder containing the input ASCII grids.
    
    Output:
    - output_file: Path to the output ASCII grid containing the per pixel average of all input grids.
    
    Raises: No. asc files found in the specified folder. if the input folder is empty or does not contain any .asc files.
    
    """
    # List all .asc files in the input folder.
    files = [f for f in os.listdir(input_folder) if f.endswith('.asc')]
    if not files:
        raise FileNotFoundError("No .asc files found in the specified folder.")
    
    # Open the first file to obtain the profile and data dimensions.
    first_file = os.path.join(input_folder, files[0])
    with rasterio.open(first_file) as src:
        profile = src.profile.copy()
        data = src.read(1)
    
    # Initialize an array to accumulate the sum.
    sum_array = np.zeros_like(data, dtype=np.float32)
    count = 0

    # Loop over each file and accumulate data.
    for file in files:
        print(f"Processing file {count+1}/{len(files)}: {file}")
        file_path = os.path.join(input_folder, file)
        with rasterio.open(file_path) as src:
            data = src.read(1)
        # Replace negative values with 0 using vectorized operation.
        data[data < 0] = 0
        sum_array += data
        count += 1

    # Calculate the average array.
    avg_array = sum_array / count

    # Update the profile for writing an ASCII grid.
    profile.update(driver="AAIGrid", dtype=rasterio.float32, count=1)
    
    # Write the output ASC file.
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(avg_array.astype(np.float32), 1)

    print(f"Output written to: {output_file}")

# Run the averaging function.
average_asc_files(input_folder, output_file)
