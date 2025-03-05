import rasterio
import numpy as np
import pandas as pd
import os
#-----------------------------------------------------------------------------------------------------------------------------------------
# Función que calcula la fracción quemada en copa (CFB)
#-----------------------------------------------------------------------------------------------------------------------------------------
def crown_fraction_burned(ros, CBH, FMC, CBD, H, fuel_load):
    """
    Calculates the crown fraction burned (cfb) for each pixel. The calculations for i0 and ros_crit are performed only on those 
    pixels where fuel_load is not zero. The formula is applied elementwise and 
    returns 0 if fuel_load is 0 or if the condition is not met.

    Inputs:
        - ros: Array of fire spread rate.
        - CBH: Array of Crown Base Height.
        - FMC: Scalar or array of Fuel Moisture Content.
        - CBD: Array of Crown Bulk Density.
        - H: Array or scalar for height (or another calibration variable).
        - fuel_load: Array of fuel load.
    
    Output:
        - cfb: Array of the crown fraction burned for each pixel.
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
    Generates an ASC raster for the crown fraction burned (CFB) using a ROS ASC file.
    It uses fixed rasters for CBH, CBD, H, and fuel load, and writes the result to output_folder.
    Returns the path of the generated ASC file.

    Parameters:
    ros_asc_path: Path to the ROS ASC file.
    cbh_raster: Path to the ASC raster for CBH.
    FMC: Scalar (or array) for Fuel Moisture Content.
    cbd_raster: Path to the ASC raster for CBD.
    h_raster: Path to the ASC raster for H.
    fuel_load_path: Path to the ASC raster for fuel load (cfl).
    output_folder: Output folder for the CFB raster.
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
    """
    Generates an ASCII raster of Heat values (H) using a lookup table and a fuel code raster.

    h_csv: Path to the CSV containing pairs of heat values and fuel codes.
    fuel_column: Name of the column containing fuel codes.
    h_column: Name of the column containing heat values.
    input_raster: Path to the raster containing fuel codes.
    output_raster: Path to save the output heat raster.

    """
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
    """
    Generates an ASCII raster of Crown Base Height (CBH) using a lookup table and a fuel code raster.

    cbh_csv: Path to the CSV containing pairs of CBH values and fuel codes.
    fuel_column: Name of the column containing fuel codes.
    cbh_column: Name of the column conatining CBH values.
    input_raster: Path to the raster containing fuel codes.
    output_raster: Path to save the output CBH raster.
    
    """
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
    """
    Generates an ASCII raster of Crown Bulk Density (CBD) using a lookup table and a fuel code raster.

    cbd_csv: Path to the CSV containing pairs of CBD values and fuel codes.
    fuel_column: Name of the column containing fuel codes.
    cbd_column: Name of the column containing CBD values.
    input_raster: Path to the raster containing fuel codes.
    output_raster: Path to save the output CBD raster.

    """
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
    """
    Generates an ASCII raster of crown fuel loads using a lookup table and a fuel code raster.

    fuel_load_csv: Path to the CSV containing pairs of fuel loads and fuel codes.
    fuel_column: Name of the column containing fuel codes.
    fuel_load_column: Name of the column containing crown fuel loads.
    input_raster: Path to the raster containing fuel codes.
    output_raster: Path to save the output crown fuel load raster.

    """
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
    Calculates the CO2 eq emisions generated by crown fire, given fuel types, crown fuel load, and crown fraction burned arrays.
    the function is vectorized si the computation can be performed for entire arrays. The function calculates fuel consumed by 
    multiplying crown fuel load by crown fraction burned and then calculates emissions for every greenhouse gas (GHG) using
    specific emission factors and a unit correction factor. Finally, a weighted sum is performed to get the CO2eq emissions.

    Inputs:
    - fuels: 2D array of fuel types (integers).
    - cfl: 2D array of crown fuel loads (floats).
    - cfb: 2D array of crown fraction burned values (floats).

    Output:
    - emisiones: 2D array of CO2eq emissions (floats).

    Sources:
    - Gas specific emission factors: IPCC Gef.
    - Weighted sum weights: IPCC GWP@100yr.

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
    Generates an ASCII emissions raster using fuels, crown fuel load, and crown fraction burned rasters.
    It calls the vectorized function to calculate the emissions elementwise and writes the result to output_folder.

    Inputs: 
    - fuels_raster_path: Path to the ASCII raster of fuel types.
    - cfl_raster: Path to the ASCII raster of crown fuel loads.
    - cfb_raster_path: Path to the ASCII raster of crown fraction burned values.
    - output_folder: Folder to save the generated emissions ASC file.

    Output:
    - output_raster: Path to the generated emissions ASC file.

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
    """
    Computes the sum of all valid pixel values in an ASCII grid file.
    
    Input:
    - raster_path: Path to the ASCII grid file (float).
    
    Output:
    - total_sum: sum of all valid pixel values in the ASCII grid file (float).
    
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        total_sum = data.sum()
    return total_sum

#-----------------------------------------------------------------------------------------------------------------------------------------
# Función que calcula el valor promedio de un raster (ignorando nodata)
#-----------------------------------------------------------------------------------------------------------------------------------------
def average_pixel_value(raster_path):
    """
    Computes the average value of all valid pixels in an Arc/Info ASCII Grid (ASC) file.
    It uses the raster's nodata value to ignore invalid pixels.

    Input:
    - raster_path: Path to the ASCII grid file (float).

    Output:
    - average: Average value of all valid pixels in the ASCII grid file (float).
    """
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
