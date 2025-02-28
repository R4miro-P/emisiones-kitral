# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
###---------------------------------------------------------------------------------------------------------------------------------------
#FUNCIONES FUEGO SUPERFICIAL
###-------------------------------------------------------------------------------------------------------------------------------------
import rasterio
import numpy as np
import os
import pandas as pd

#--------------------------------------------------------------------------------------------------------------------------------
#se crea funcion que calcula sfb a partir de Ros.
def surface_fuel_consumed_vectorized(fuels, ros):
    
    #Calcula la fraccion de combustible consumido en superficie a partir de los raster de combustible y ROS
    
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
#--------------------------------------------------------------------------------------------------------------------------------------
#Se crea una funcion que recibe un raster de ros y uno de cargas y calcula la fraccion consumida en superficie. Notar si ROS=0 sfb=0
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
#-----------------------------------------------------------------------------------------------------------------------------------------
#Se crea un raster de Fuel Load en Superficie
def generate_fuel_load_raster():

    # 1. Read the CSV containing (fuel code -> fuel load)
    fuel_load_csv = "/home/ramiro/Emisiones/lookup_ramiro.csv"
    fuel_column = "Fuel Code"
    fuel_load_column = "fl"

    df_fuel_load = pd.read_csv(fuel_load_csv, sep=";")
    print(df_fuel_load.columns)

    # Create a dictionary {fuel_code: fuel_load}
    code_to_fuel_load = dict(zip(df_fuel_load[fuel_column],
                                df_fuel_load[fuel_load_column]))

    # 2. Paths to input (fuel-code) raster and output (fuel-load) ASCII
    input_raster = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"
    output_raster = "/home/ramiro/Emisiones/fuel_load.asc"

    # 3. Open the input raster
    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)      # Read the first band as a NumPy array
        profile = src.profile.copy()   # Copy the metadata (profile)

    # 4. Create an empty array (float32) to store fuel loads
    fuel_load_raster = np.zeros_like(raster_data, dtype=np.float32)

    # 5. Replace each fuel code with the corresponding fuel load
    for code, fuel_load in code_to_fuel_load.items():
        # Where the raster_data equals "code", set the output to the fuel load
        fuel_load_raster[raster_data == code] = fuel_load

    # 6. Update the profile for ASCII output
    #    - Specify driver='AAIGrid' so rasterio writes Arc/Info ASCII
    #    - Make sure dtype and nodata are set appropriately
    profile.update(
        driver="AAIGrid",
        dtype=rasterio.float32,
        nodata=0
    )

    # 7. Write the output ASCII grid
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(fuel_load_raster, 1)

    print(f"ASCII grid with fuel loads created: {output_raster}")
    return output_raster
#-----------------------------------------------------------------------------------------------------------------------------------------
#Funcion que calcula las emisiones generadas
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
    eCO2 = 1613 * fuel_consumed[idx1] * 10**(-2)
    eCH4 = 2.3 * fuel_consumed[idx1] * 10**(-2)
    eN2O = 0.21 * fuel_consumed[idx1] * 10**(-2)
    emisiones[idx1] = eCO2 + eCH4 * 27 + eN2O * 273

    # Condiciones para Matorrales: (fuels > 5) y (fuels < 14)
    idx2 = (fuels > 5) & (fuels < 14)
    eCO2 = 1613 * fuel_consumed[idx2] * 10**(-2)
    eCH4 = 2.3 * fuel_consumed[idx2] * 10**(-2)
    eN2O = 0.21 * fuel_consumed[idx2] * 10**(-2)
    emisiones[idx2] = eCO2 + eCH4 * 27 + eN2O * 273

    # Condiciones para Arboles: (fuels > 13) y (fuels < 18)
    idx3 = (fuels > 13) & (fuels < 18)
    eCO2 = 1569 * fuel_consumed[idx3] * 10**(-2)
    eCH4 = 4.7 * fuel_consumed[idx3] * 10**(-2)
    eN2O = 0.26 * fuel_consumed[idx3] * 10**(-2)
    emisiones[idx3] = eCO2 + eCH4 * 27 + eN2O * 273

    # Condiciones para Arboles de plantación: fuels > 17
    idx4 = fuels > 17
    eCO2 = 1569 * fuel_consumed[idx4] * 10**(-2)
    eCH4 = 4.7 * fuel_consumed[idx4] * 10**(-2)
    eN2O = 0.26 * fuel_consumed[idx4] * 10**(-2)
    emisiones[idx4] = eCO2 + eCH4 * 27 + eN2O * 273

    return emisiones
#-----------------------------------------------------------------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------------------------------------------------------
#Se crea una funcion que suma el total de emisiones generadas.
def sum_raster_values(raster_path):
    
    #Suma los valores de cada pixel en un raster (ignora los nodata si estan masked).
    
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        total_sum = data.sum()
    return total_sum
#-----------------------------------------------------------------------------------------------------------------------------------------
#Funcion que calcula valor promedio de consumo
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
#-----------------------------------------------------------------------------------------------------------------------------------------

# %%
# -------------------------------------------------------------------
# MASTER MAIN FUNCTION FUEGO SUPERFICIAL
# -------------------------------------------------------------------


def main():
    # Generate the fuel load raster once (assumed constant).
    fuel_load_raster = generate_fuel_load_raster()  # This function must return the output file path.
    
    # Fixed fuels raster path.
    fuels_raster_path = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"
    
    # Folder containing the ROS ASC files.
    ros_folder = "/home/ramiro/Emisiones/50sim/ROS/"
    # Folder to store generated SFB rasters.
    sfb_output_folder = "/home/ramiro/Emisiones/50sim/SFB/"
    # Folder to store geneated emissions rasters.
    emissions_output_folder = "/home/ramiro/Emisiones/50sim/Emisiones/"
    
    # List all ROS ASC files in the folder.
    ros_files = [os.path.join(ros_folder, f) for f in os.listdir(ros_folder) if f.endswith('.asc')]
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
    
    print("\nList of emissions sums: ",emisiones_sums)
    print("\nlist of average pixel values: ",average_value)

# -------------------------------------------------------------------
# Execute master main if run as a script
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()


# %%
###---------------------------------------------------------------------------------------------------------------------------------------
# CALCULO DE SUM(pixel values)/Nsims
###---------------------------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import rasterio

input_folder = "/home/ramiro/Emisiones/50sim/SFB/"
output_file = "/home/ramiro/Emisiones/50sim/cicatriz_total.asc" 

def average_asc_files(input_folder, output_file):
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


# %%
###---------------------------------------------------------------------------------------------------------------------------------------
# CALCULO DE VALORES PROMEDIO POR PIXEL PARA EL TOTAL DE TODOS LOS INCENDIOS DE SUPERFICIE
###---------------------------------------------------------------------------------------------------------------------------------------
import rasterio
import numpy as np

def average_pixel_value_total(asc_file, bp_file):
    """
    Computes the average value of all valid pixels in an Arc/Info ASCII Grid (ASC) file.
    It uses the raster's nodata value to ignore invalid pixels.
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


# %%
import os
import numpy as np
import rasterio
import pandas as pd

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función que calcula la fracción quemada en copa (CFB)
#-----------------------------------------------------------------------------------------------------------------------------------------
def crown_fraction_burned(ros, CBH, FMC, CBD, H, fuel_load):
    """
    Calcula la fracción quemada en copa (cfb) para cada píxel. Se realizan los cálculos de i0 y ros_crit 
    solo en aquellos píxeles donde fuel_load es distinto de 0. La fórmula se aplica de forma elementwise y se 
    retorna 0 en caso de que fuel_load sea 0 o si no se cumple la condición.
    
    Parámetros:
      ros: Array de tasa de propagación del fuego.
      CBH: Array de Crown Base Height.
      FMC: Escalar o array de Fuel Moisture Content.
      CBD: Array de Crown Bulk Density.
      H: Array o escalar para la altura (u otra variable de calibración).
      fuel_load: Array de carga de combustible.
      
    Retorna:
      cfb: Array de la fracción quemada en copa para cada píxel.
    """
    # Calcular i0 solo donde fuel_load es distinto de 0; de lo contrario, 0.
    i0 = np.where((fuel_load != 0) & (H != 0), 0.01 * CBH * (460 + 25.9 * FMC)**(1.5), 0)
    # Calcular ros_crit solo donde fuel_load es distinto de 0; de lo contrario, 0.
    valid = (fuel_load != 0) & (H != 0)
    ros_crit = np.zeros_like(i0)
    # Compute division only where valid.
    ros_crit[valid] = 60 * i0[valid] / (H[valid] * fuel_load[valid])
    
    # Calcular cfb solo donde fuel_load != 0, ros_crit > ros y CBD != 0.
    valid2 = (ros != 0) & (ros_crit > ros) & (CBD != 0)
    cfb = np.zeros_like(i0)
    cfb[valid2] = 1 - np.exp(- (-np.log10(0.1) / (0.9 * ((3/CBD[valid2]) - ros_crit[valid2]))) * (ros[valid2] - ros_crit[valid2]))
    return cfb

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función que genera el raster ASC de fracción quemada en copa (CFB)
#-----------------------------------------------------------------------------------------------------------------------------------------
def generate_crown_fraction_burned(ros_asc_path, cbh_raster, FMC, cbd_raster, h_raster, fuel_load_path, output_folder):
    """
    Genera un raster ASC para la fracción quemada en copa (CFB) usando un archivo ROS ASC.
    Utiliza rasters fijos para CBH, CBD, H y la carga de combustible, y escribe el resultado en output_folder.
    Retorna el path del archivo ASC generado.
    
    Parámetros:
      ros_asc_path: Path al archivo ROS ASC.
      cbh_raster: Path al raster ASC de CBH.
      FMC: Escalar (o array) de Fuel Moisture Content.
      cbd_raster: Path al raster ASC de CBD.
      h_raster: Path al raster ASC de H.
      fuel_load_path: Path al raster ASC de la carga de combustible (cfl).
      output_folder: Carpeta de salida para el raster CFB.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ros_basename = os.path.basename(ros_asc_path)
    output_asc = os.path.join(output_folder, f"crown_fraction_burned_{ros_basename}")
    
    # Leer los rasters fijos.
    with rasterio.open(cbh_raster) as src:
        cbh = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(cbd_raster) as src:
        cbd = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(h_raster) as src:
        h = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(fuel_load_path) as src:
        fuel_load = src.read(1)
        profile = src.profile.copy()
    
    # Leer el raster ROS.
    with rasterio.open(ros_asc_path) as src:
        ros = src.read(1)
    
    # Opcional: Verificar dimensiones (se asume que todos los rasters tienen la misma dimensión)
    if cbh.shape != ros.shape:
        raise ValueError(f"Los rasters CBH y ROS deben tener las mismas dimensiones. Got {cbh.shape} y {ros.shape}.")
    
    # Importante: Llamar a crown_fraction_burned con el orden correcto:
    # (ros, CBH, FMC, CBD, H, fuel_load)
    cfb_array = crown_fraction_burned(ros, cbh, FMC, cbd, h, fuel_load)
    
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0)
    
    with rasterio.open(output_asc, "w", **profile) as dst:
        dst.write(cfb_array, 1)
    
    print(f"CFB raster saved to: {output_asc}")
    return output_asc

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función para generar el raster de H
#-----------------------------------------------------------------------------------------------------------------------------------------
def generate_h_raster():
    h_csv = "/home/ramiro/Emisiones/lookup_ramiro.csv"
    fuel_column = 'Fuel Code'
    h_column = 'h'
    df_h = pd.read_csv(h_csv, sep=';')
    df_h.columns = df_h.columns.str.strip()
    fuel_to_h = dict(zip(df_h[fuel_column], df_h[h_column]))
    print("H CSV columns:", df_h.columns.tolist())
    input_raster = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"
    output_raster = "/home/ramiro/Emisiones/50sim/h.asc"
    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)
        profile = src.profile.copy()
    h_raster = np.zeros_like(raster_data, dtype=np.float32)
    for fuel, h in fuel_to_h.items():
        h_raster[raster_data == fuel] = h
    profile.update(driver='AAIGrid', dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(h_raster.astype(np.float32), 1)
    print(f"H raster saved to: {output_raster}")
    return output_raster

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función para generar el raster de CBH
#-----------------------------------------------------------------------------------------------------------------------------------------
def generate_cbh_raster():
    cbh_csv = "/home/ramiro/Emisiones/lookup_ramiro.csv"
    fuel_column = 'Fuel Code'
    cbh_column = 'cbh'
    df_cbh = pd.read_csv(cbh_csv, sep=';')
    df_cbh.columns = df_cbh.columns.str.strip()
    fuel_to_cbh = dict(zip(df_cbh[fuel_column], df_cbh[cbh_column]))
    input_raster = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"
    output_raster = "/home/ramiro/Emisiones/50sim/cbh.asc"
    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)
        profile = src.profile.copy()
    cbh_raster = np.zeros_like(raster_data, dtype=np.float32)
    for fuel, cbh in fuel_to_cbh.items():
        cbh_raster[raster_data == fuel] = cbh
    profile.update(driver='AAIGrid', dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(cbh_raster.astype(np.float32), 1)
    print(f"CBH raster saved to: {output_raster}")
    return output_raster

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función para generar el raster de CBD
#-----------------------------------------------------------------------------------------------------------------------------------------
def generate_cbd_raster():
    cbd_csv = "/home/ramiro/Emisiones/lookup_ramiro.csv"
    fuel_column = 'Fuel Code'
    cbd_column = 'cbd'
    df_cbd = pd.read_csv(cbd_csv, sep=';')
    df_cbd.columns = df_cbd.columns.str.strip()
    fuel_to_cbd = dict(zip(df_cbd[fuel_column], df_cbd[cbd_column]))
    print("CBD CSV columns:", df_cbd.columns.tolist())
    input_raster = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"
    output_raster = "/home/ramiro/Emisiones/50sim/cbd.asc"
    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)
        profile = src.profile.copy()
    cbd_raster = np.zeros_like(raster_data, dtype=np.float32)
    for fuel, cbd in fuel_to_cbd.items():
        cbd_raster[raster_data == fuel] = cbd
    profile.update(driver='AAIGrid', dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(cbd_raster.astype(np.float32), 1)
    print(f"CBD raster saved to: {output_raster}")
    return output_raster

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función para generar el raster de carga de combustible en copa (cfl)
#-----------------------------------------------------------------------------------------------------------------------------------------
def generate_crown_fuel_load_raster():
    fuel_load_csv = "/home/ramiro/Emisiones/lookup_ramiro.csv"
    fuel_column = "Fuel Code"
    fuel_load_column = "cfl"
    df_fuel_load = pd.read_csv(fuel_load_csv, sep=";")
    print(df_fuel_load.columns)
    code_to_fuel_load = dict(zip(df_fuel_load[fuel_column],
                                df_fuel_load[fuel_load_column]))
    input_raster = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"
    output_raster = "/home/ramiro/Emisiones/50sim/cfl.asc"
    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)
        profile = src.profile.copy()
    fuel_load_raster = np.zeros_like(raster_data, dtype=np.float32)
    for code, fuel_load in code_to_fuel_load.items():
        fuel_load_raster[raster_data == code] = fuel_load
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0)
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(fuel_load_raster, 1)
    print(f"ASCII grid with fuel loads created: {output_raster}")
    return output_raster

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función que calcula las emisiones generadas por incendio de copa (vectorizada)
#-----------------------------------------------------------------------------------------------------------------------------------------
def emisiones_generadas_vectorized(fuels, cfl, cfb):
    """
    Calcula las emisiones generadas (en CO2eq) para cada píxel a partir
    de los arrays: combustibles (fuels), carga de combustible (cfl) y
    la fracción quemada (cfb). Se espera que todos tengan la misma dimensión.
    """
    if fuels.shape != cfl.shape or fuels.shape != cfb.shape:
        raise ValueError("All input arrays must have the same dimensions.")
    fuel_consumed = cfl * cfb
    emisiones = np.zeros_like(fuels, dtype=np.float32)
    # Condiciones para Pastizales: (fuels > 0) y (fuels < 6)
    idx1 = (fuels > 0) & (fuels < 6)
    eCO2 = 1613 * fuel_consumed[idx1] * 10**(-2)
    eCH4 = 2.3 * fuel_consumed[idx1] * 10**(-2)
    eN2O = 0.21 * fuel_consumed[idx1] * 10**(-2)
    emisiones[idx1] = eCO2 + eCH4 * 27 + eN2O * 273
    # Condiciones para Matorrales: (fuels > 5) y (fuels < 14)
    idx2 = (fuels > 5) & (fuels < 14)
    eCO2 = 1613 * fuel_consumed[idx2] * 10**(-2)
    eCH4 = 2.3 * fuel_consumed[idx2] * 10**(-2)
    eN2O = 0.21 * fuel_consumed[idx2] * 10**(-2)
    emisiones[idx2] = eCO2 + eCH4 * 27 + eN2O * 273
    # Condiciones para Árboles: (fuels > 13) y (fuels < 18)
    idx3 = (fuels > 13) & (fuels < 18)
    eCO2 = 1569 * fuel_consumed[idx3] * 10**(-2)
    eCH4 = 4.7 * fuel_consumed[idx3] * 10**(-2)
    eN2O = 0.26 * fuel_consumed[idx3] * 10**(-2)
    emisiones[idx3] = eCO2 + eCH4 * 27 + eN2O * 273
    # Condiciones para Árboles de plantación: fuels > 17
    idx4 = fuels > 17
    eCO2 = 1569 * fuel_consumed[idx4] * 10**(-2)
    eCH4 = 4.7 * fuel_consumed[idx4] * 10**(-2)
    eN2O = 0.26 * fuel_consumed[idx4] * 10**(-2)
    emisiones[idx4] = eCO2 + eCH4 * 27 + eN2O * 273
    return emisiones

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función que genera el raster ASC de emisiones generadas
#-----------------------------------------------------------------------------------------------------------------------------------------
def generate_emisiones_generadas_raster(fuels_raster_path, cfl_raster, cfb_raster_path, output_folder):
    """
    Genera un raster ASC de emisiones generadas usando los rasters de combustibles, 
    carga de combustible en copa (cfl) y fracción quemada (cfb). Guarda el resultado 
    en output_folder con un nombre basado en el raster cfb.
    Retorna el path del raster generado.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cfb_basename = os.path.basename(cfb_raster_path)
    name, _ = os.path.splitext(cfb_basename)
    output_raster = os.path.join(output_folder, f"emisiones_generadas_{name}.asc")
    
    with rasterio.open(fuels_raster_path) as src:
        fuels = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(cfl_raster) as src:
        cfl = src.read(1)
    with rasterio.open(cfb_raster_path) as src:
        cfb = src.read(1)
    
    if fuels.shape != cfl.shape or fuels.shape != cfb.shape:
        raise ValueError("The rasters (fuels, crown fuel load, and cfb) must have the same dimensions.")
    
    emisiones_array = emisiones_generadas_vectorized(fuels, cfl, cfb)
    
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0)
    
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(emisiones_array, 1)
    
    print(f"Emisiones generadas raster saved to: {output_raster}")
    return output_raster

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función que suma los valores de un raster (ignorando nodata)
#-----------------------------------------------------------------------------------------------------------------------------------------
def sum_raster_values(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        total_sum = data.sum()
    return total_sum

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función que calcula el valor promedio de un raster (ignorando nodata)
#-----------------------------------------------------------------------------------------------------------------------------------------
def average_pixel_value(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        average = np.ma.mean(data)
    return average


# %%
#-----------------------------------------------------------------------------------------------------------------------------------------
# MASTER MAIN FUNCTION FUEGO DE COPA
#-----------------------------------------------------------------------------------------------------------------------------------------
def main():
    # Generar rasters fijos
    fuel_load_raster = generate_crown_fuel_load_raster()  # cfl raster
    cbd_raster = generate_cbd_raster()
    cbh_raster = generate_cbh_raster()
    h_raster = generate_h_raster()
    
    # Ruta fija al raster de combustibles.
    fuels_raster_path = "/home/ramiro/on-boarding/kitral/Kitral/Portillo-asc/fuels.asc"
    
    # Carpetas de entrada y salida.
    cfb_folder = "/home/ramiro/Emisiones/50sim/CFB_manual/"
    ros_folder = "/home/ramiro/Emisiones/50sim/ROS/"
    emissions_output_folder = "/home/ramiro/Emisiones/50sim/Emisiones_copa/"
    
    # Listar todos los archivos ROS ASC en la carpeta.
    ros_files = [os.path.join(ros_folder, f) for f in os.listdir(ros_folder) if f.endswith('.asc')]
    if not ros_files:
        raise FileNotFoundError("No ROS ASC files found in the specified folder.")
    
    emisiones_sums = []
    average_value = []
    
    # Procesar cada archivo ROS.
    for ros_file in ros_files:
        print(f"\nProcessing ROS file: {ros_file}")
        
        # Generar raster CFB usando el archivo ROS y rasters fijos.
        # Nota: El parámetro FMC se pasa como 40 (constante); asegúrese de que este valor sea el deseado.
        cfb_raster = generate_crown_fraction_burned(ros_file, cbh_raster, 66, cbd_raster, h_raster, fuel_load_raster, cfb_folder)
        
        # Generar raster de emisiones usando el raster de combustibles, el raster de carga de combustible (cfl)
        # y el raster CFB generado.
        emisiones_generadas_raster = generate_emisiones_generadas_raster(fuels_raster_path, fuel_load_raster, cfb_raster, emissions_output_folder)
        
        # Sumar los valores del raster de emisiones.
        emis_sum = sum_raster_values(emisiones_generadas_raster)
        # Calcular el valor promedio del raster CFB.
        avg_val = average_pixel_value(cfb_raster)
        
        emisiones_sums.append(emis_sum)
        average_value.append(avg_val)
        
        print(f"Sum of emisiones_generadas for {os.path.basename(ros_file)}: {emis_sum}")
        print(f"Average pixel value for {os.path.basename(ros_file)}: {avg_val}")
    
    print("\nList of emissions sums:", emisiones_sums)
    print("\nList of average pixel values:", average_value)

if __name__ == "__main__":
    main()

# %%
###---------------------------------------------------------------------------------------------------------------------------------------
# CALCULO DE SUM(pixel values)/Nsims PARA FUEGO DE COPA
###---------------------------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import rasterio

input_folder = "/home/ramiro/Emisiones/50sim/CFB_manual/"
output_file = "/home/ramiro/Emisiones/50sim/cicatriz_total_copa.asc" 

def average_asc_files(input_folder, output_file):
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


# %%
###---------------------------------------------------------------------------------------------------------------------------------------
# CALCULO DE VALORES PROMEDIO POR PIXEL PARA EL TOTAL DE TODOS LOS INCENDIOS DE COPA
###---------------------------------------------------------------------------------------------------------------------------------------
import rasterio
import numpy as np

def average_pixel_value_total(asc_file, bp_file):
    """
    Computes the average value of all valid pixels in an Arc/Info ASCII Grid (ASC) file.
    It uses the raster's nodata value to ignore invalid pixels.
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
asc_file = "/home/ramiro/Emisiones/50sim/cicatriz_total_copa.asc"  # Replace with your actual ASC file path
bp_file = "/home/ramiro/Emisiones/50sim/bp.asc"
avg_value_total = average_pixel_value_total(asc_file, bp_file)
print("Average pixel value:", avg_value_total)

