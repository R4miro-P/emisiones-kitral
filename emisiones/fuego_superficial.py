###---------------------------------------------------------------------------------------------------------------------------------------
# FUNCIONES FUEGO SUPERFICIAL
###-------------------------------------------------------------------------------------------------------------------------------------
import os

import numpy as np
import pandas as pd
import rasterio


def breakit():
    # fmt: off
    from IPython.terminal.embed import InteractiveShellEmbed

    # for QGIS
    # from qgis.PyQt.QtCore import pyqtRemoveInputHook  # type: ignore
    # pyqtRemoveInputHook()
    # fmt: on
    return InteractiveShellEmbed()


# --------------------------------------------------------------------------------------------------------------------------------
# se crea funcion que calcula sfb a partir de Ros.
def surface_fuel_consumed_vectorized(fuels, ros):

    # Calcula la fraccion de combustible consumido en superficie a partir de los raster de combustible y ROS

    sfc = np.zeros_like(fuels, dtype=np.float32)
    # Condition 1: 0 < fuels < 6
    idx1 = (fuels > 0) & (fuels < 6)
    sfc[idx1] = 1 - np.exp(-0.15 * ros[idx1])
    # Condition 2: 5 < fuels < 14
    idx2 = (fuels > 5) & (fuels < 14)
    sfc[idx2] = 1 - np.exp(-0.1 * ros[idx2])
    # Condition 3: 13 < fuels < 18
    idx3 = (fuels > 13) & (fuels < 18)
    sfc[idx3] = 1 - np.exp(-0.1 * ros[idx3])
    # Condition 4: fuels > 17
    idx4 = fuels > 17
    sfc[idx4] = 1 - np.exp(-0.06 * ros[idx4])
    return sfc


# --------------------------------------------------------------------------------------------------------------------------------------
# Se crea una funcion que recibe un raster de ros y uno de cargas y calcula la fraccion consumida en superficie. Notar si ROS=0 sfb=0
def generate_surface_fraction_burned(ros_asc_path, output_folder):
    """
    Generates an ASCII grid for surface fraction burned (SFB) using a given ROS ASC file.
    Uses a fixed fuels raster and writes the output to output_folder.
    Returns the path to the generated SFB ASC file.
    """
    fuels_raster = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct an output file name based on the ROS file name.
    ros_basename = os.path.basename(ros_asc_path)
    output_asc = os.path.join(output_folder, f"surface_fraction_burned_{ros_basename}")

    # Read the fuels raster.
    with rasterio.open(fuels_raster) as src:
        fuels = src.read(1)
        profile = src.profile.copy()

    # Read the current ROS file.
    with rasterio.open(ros_asc_path) as src:
        ros = src.read(1)

    if fuels.shape != ros.shape:
        raise ValueError(f"Fuels and ROS rasters must have the same dimensions. Got {fuels.shape} and {ros.shape}.")

    # Calculate the SFB array using your vectorized function (assumed defined elsewhere).
    sfc_array = surface_fuel_consumed_vectorized(fuels, ros)

    # Update profile for ASCII grid output.
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0)

    # Write out the SFB ASC file.
    with rasterio.open(output_asc, "w", **profile) as dst:
        dst.write(sfc_array, 1)

    print(f"SFB raster saved to: {output_asc}")
    return output_asc


# -----------------------------------------------------------------------------------------------------------------------------------------
# Se crea un raster de Fuel Load en Superficie
def generate_fuel_load_raster(*args, **kwargs):

    # 1. Read the CSV containing (fuel code -> fuel load)
    fuel_load_csv = "/home/ramiro/Emisiones/lookup_ramiro.csv"
    fuel_column = "Fuel Code"
    fuel_load_column = "fl"

    breakit()()
    df_fuel_load = pd.read_csv(fuel_load_csv, sep=";")
    print(df_fuel_load.columns)

    # Create a dictionary {fuel_code: fuel_load}
    code_to_fuel_load = dict(zip(df_fuel_load[fuel_column], df_fuel_load[fuel_load_column]))

    # 2. Paths to input (fuel-code) raster and output (fuel-load) ASCII
    input_raster = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"
    output_raster = "/home/ramiro/Emisiones/fuel_load.asc"

    # 3. Open the input raster
    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)  # Read the first band as a NumPy array
        profile = src.profile.copy()  # Copy the metadata (profile)

    # 4. Create an empty array (float32) to store fuel loads
    fuel_load_raster = np.zeros_like(raster_data, dtype=np.float32)

    # 5. Replace each fuel code with the corresponding fuel load
    for code, fuel_load in code_to_fuel_load.items():
        # Where the raster_data equals "code", set the output to the fuel load
        fuel_load_raster[raster_data == code] = fuel_load

    # 6. Update the profile for ASCII output
    #    - Specify driver='AAIGrid' so rasterio writes Arc/Info ASCII
    #    - Make sure dtype and nodata are set appropriately
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0)

    # 7. Write the output ASCII grid
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(fuel_load_raster, 1)

    print(f"ASCII grid with fuel loads created: {output_raster}")
    return output_raster


# -----------------------------------------------------------------------------------------------------------------------------------------
# Funcion que calcula las emisiones generadas
def emisiones_generadas_vectorized(fuels, fuel_load, sfb):
    """
    Calcula las emisiones generadas (en CO2eq) para cada pixel a partir
    de los arrays: combustibles (fuels), carga de combustible (fuel_load) y
    la fracción quemada (sfb). Se espera que todos tengan la misma dimensión.
    """
    # Optional: check that all arrays have the same shape
    if fuels.shape != fuel_load.shape or fuels.shape != sfb.shape:
        raise ValueError("All input arrays must have the same dimensions.")

    # Calcular la carga consumida
    fuel_consumed = fuel_load * sfb

    # Inicializar el array de emisiones con ceros (mismo shape que fuels)
    emisiones = np.zeros_like(fuels, dtype=np.float32)

    # Condiciones para Pastizales: (fuels > 0) y (fuels < 6)
    idx1 = (fuels > 0) & (fuels < 6)
    eCO2 = 1613 * fuel_consumed[idx1] * 10 ** (-2)
    eCH4 = 2.3 * fuel_consumed[idx1] * 10 ** (-2)
    eN2O = 0.21 * fuel_consumed[idx1] * 10 ** (-2)
    emisiones[idx1] = eCO2 + eCH4 * 27 + eN2O * 273

    # Condiciones para Matorrales: (fuels > 5) y (fuels < 14)
    idx2 = (fuels > 5) & (fuels < 14)
    eCO2 = 1613 * fuel_consumed[idx2] * 10 ** (-2)
    eCH4 = 2.3 * fuel_consumed[idx2] * 10 ** (-2)
    eN2O = 0.21 * fuel_consumed[idx2] * 10 ** (-2)
    emisiones[idx2] = eCO2 + eCH4 * 27 + eN2O * 273

    # Condiciones para Arboles: (fuels > 13) y (fuels < 18)
    idx3 = (fuels > 13) & (fuels < 18)
    eCO2 = 1569 * fuel_consumed[idx3] * 10 ** (-2)
    eCH4 = 4.7 * fuel_consumed[idx3] * 10 ** (-2)
    eN2O = 0.26 * fuel_consumed[idx3] * 10 ** (-2)
    emisiones[idx3] = eCO2 + eCH4 * 27 + eN2O * 273

    # Condiciones para Arboles de plantación: fuels > 17
    idx4 = fuels > 17
    eCO2 = 1569 * fuel_consumed[idx4] * 10 ** (-2)
    eCH4 = 4.7 * fuel_consumed[idx4] * 10 ** (-2)
    eN2O = 0.26 * fuel_consumed[idx4] * 10 ** (-2)
    emisiones[idx4] = eCO2 + eCH4 * 27 + eN2O * 273

    return emisiones


# -----------------------------------------------------------------------------------------------------------------------------------------
def generate_emisiones_generadas_raster(fuels_raster_path, fuel_load_raster, sfb_raster_path, output_folder):
    """
    Generates an ASCII emissions raster using the fuels, fuel load, and SFB rasters.
    Saves the output ASC file to the specified output folder with a unique name.
    Returns the path to the generated emissions ASC file.
    """
    # Ensure output folder exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use the base name of the SFB file to construct a unique emissions file name.
    sfb_basename = os.path.basename(sfb_raster_path)
    name, _ = os.path.splitext(sfb_basename)
    output_raster = os.path.join(output_folder, f"emisiones_generadas_{name}.asc")

    # Read fuels raster.
    with rasterio.open(fuels_raster_path) as src:
        fuels = src.read(1)
        profile = src.profile.copy()

    # Read fuel load raster.
    with rasterio.open(fuel_load_raster) as src:
        fuel_load = src.read(1)

    # Read SFB raster.
    with rasterio.open(sfb_raster_path) as src:
        sfb = src.read(1)

    # Verify dimensions match.
    if fuels.shape != fuel_load.shape or fuels.shape != sfb.shape:
        raise ValueError("The rasters (fuels, fuel load, and sfb) must have the same dimensions.")

    # Calculate emissions using the vectorized function.
    emisiones_array = emisiones_generadas_vectorized(fuels, fuel_load, sfb)

    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0)

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(emisiones_array, 1)

    print(f"Emisiones generadas raster saved to: {output_raster}")
    return output_raster


# -----------------------------------------------------------------------------------------------------------------------------------------
# Se crea una funcion que suma el total de emisiones generadas.
def sum_raster_values(raster_path):

    # Suma los valores de cada pixel en un raster (ignora los nodata si estan masked).

    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        total_sum = data.sum()
    return total_sum


# -----------------------------------------------------------------------------------------------------------------------------------------
# Funcion que calcula valor promedio de consumo
def average_pixel_value(raster_path):
    """
    Computes the average value of all valid pixels in an Arc/Info ASCII Grid (ASC) file.
    It uses the raster's nodata value to ignore invalid pixels.
    """
    with rasterio.open(raster_path) as src:
        # Read the first band as a masked array so that nodata values are ignored.
        data = src.read(1, masked=True)
        # Calculate the mean of the valid (non-masked) values.
        average = np.ma.mean(data)
    return average


# -----------------------------------------------------------------------------------------------------------------------------------------

# %%
# -------------------------------------------------------------------
# MASTER MAIN FUNCTION FUEGO SUPERFICIAL
# -------------------------------------------------------------------


def main():
    # Generate the fuel load raster once (assumed constant).
    fuel_load_raster = generate_fuel_load_raster(hola="hey")  # This function must return the output file path.

    # Fixed fuels raster path.
    fuels_raster_path = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"

    # Folder containing the ROS ASC files.
    ros_folder = "/home/ramiro/Emisiones/50sim/ROS/"
    # Folder to store generated SFB rasters.
    sfb_output_folder = "/home/ramiro/Emisiones/50sim/SFB/"
    # Folder to store geneated emissions rasters.
    emissions_output_folder = "/home/ramiro/Emisiones/50sim/Emisiones/"

    # List all ROS ASC files in the folder.
    ros_files = [os.path.join(ros_folder, f) for f in os.listdir(ros_folder) if f.endswith(".asc")]
    if not ros_files:
        raise FileNotFoundError("No ROS ASC files found in the specified folder.")

    # Lists to store emissions raster paths and the corresponding sum values.
    emisiones_sums = []
    average_value = []

    # Loop over each ROS file.
    for ros_file in ros_files:
        print(f"\nProcessing ROS file: {ros_file}")

        # Step 1: Generate surface fraction burned raster for the current ROS file.
        sfb_raster = generate_surface_fraction_burned(ros_file, sfb_output_folder)

        # Step 2: Generate emissions raster using fuels.asc, fuel load, and the current SFB raster.
        emisiones_generadas_raster = generate_emisiones_generadas_raster(
            fuels_raster_path, fuel_load_raster, sfb_raster, emissions_output_folder
        )

        # Step 3: Sum the emissions raster values.
        emis_sum = sum_raster_values(emisiones_generadas_raster)

        avg_value = average_pixel_value(sfb_raster)

        # Store the results.
        emisiones_sums.append(emis_sum)
        average_value.append(avg_value)

        print(f"Sum of emisiones_generadas for {os.path.basename(ros_file)}: {emis_sum}")
        print(f"Average pixel value for {os.path.basename(ros_file)}: {avg_value}")

    print("\nList of emissions sums: ", emisiones_sums)
    print("\nlist of average pixel values: ", average_value)


# -------------------------------------------------------------------
# Execute master main if run as a script
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
