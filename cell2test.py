#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# pipeline.py — End-to-end pipeline with SFB-from-ROS, BP, DPV, optimization,
#                and validation for multiple treatment alphas.
# Flux:
# 1  Simulate (Cell2Fire)
# 2  Get ROS and compute SFB
# 3  Compute BP
# 4  Compute SFB mean per burned pixel (conditional)
# 5  Compute per-pixel max emissions
# 6  Multiply SFB_cond by max emissions -> expected emissions raster
# 7  Build DPV using expected emissions as risk
# 8  Optimize over DPV
# 9  Write treatment rasters
# 10 Validate again with Cell2Fire using treatment
# 11 Repeat steps 2–6 on validation outputs
# 12 Stats & plots (with 95% CI) and harvest emissions added
# =============================================================================

import os
import sys
import glob
import shutil
import argparse
import numpy as np
import rasterio
import subprocess as subp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# External helpers you already have:
from gurobipy import GRB
import gurobipy as gp
from operations_raster import read_asc, raster_to_dict, write_asc
from operations_msg import get_all_messages, harvested

# -----------------------------
# Config helpers
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def default_config(forest: str):
    root = f"/home/ramiro/Forest/{forest}"
    cfg = {
        "forest": forest,
        "paths": {
            "instance_root": f"{root}/instance",
            "output_preset": f"{root}/output/preset",
            "results_root":  f"{root}/results",
            "validation_root": f"{root}/output/results/validation",

            # Inputs
            "fuels_asc": f"{root}/instance/fuels.asc",
            # Provide your surface fuel load raster here:
            "fl_sup_asc": f"/home/ramiro/rasters/fuel_load_{forest}.asc",

            # Derived folders/files (preset)
            "ros_preset": f"{root}/output/preset/RateOfSpread",
            "sfb_preset": f"{root}/results/preset/SFB",
            "mean_sfb_preset": f"{root}/results/preset/mean_sfb.asc",
            "bp_preset": f"{root}/results/preset/bp.asc",
            "sfb_cond_preset": f"{root}/results/preset/mean_sfb_cond.asc",
            "emis_max_preset": f"{root}/results/preset/emisiones_max.asc",
            "expected_emis_preset": f"{root}/results/preset/expected_emis.asc",
            "messages_preset": f"{root}/output/preset/Messages",
            "dpv_preset": f"{root}/results/preset/dpv.asc",

            # Harvest
            "harvest_dir": f"{root}/results/harvest",
        },
        "sim": {
            "nsims_preset": 10,
            "nsims_val": 60000,
            "nthreads": 7,
            "seed_preset": 123,
            "seed_val": 333,
            "weather_preset": "random",
            "weather_val": "replication",
        },
        # Will be overridden from CLI
        "opt": {
            "alphas": [0.0, 0.03, 0.05, 0.06, 0.07, 0.6],
            # objective: "pure" uses DPV only; "dpv_minus_penalty" subtracts emis_max
            "objective": "dpv_minus_penalty",
        }
    }
    return cfg


# -----------------------------
# (1) Simulate: Cell2Fire preset
# -----------------------------
def c2f_preset(forest, instance_root, output_preset, nsims, nthreads, seed, weather):
    input_folder = f'--input-instance-folder {instance_root}'
    output_folder = f'--output-folder {output_preset}'
    nsims_opt = f'--nsims {nsims}'
    nthreads_opt = f'--nthreads {nthreads}'
    weather_opt = f'--nweathers 50 --weather {weather}'
    seed_opt = f'--seed {seed}'
    extra = '--cros'
    outputs = '--out-ros --output-messages --out-crown'
    cmd = '/home/ramiro/C2F-W/Cell2Fire/Cell2Fire --sim K --ignitions-random ' + \
          " ".join([input_folder, output_folder, nsims_opt, nthreads_opt, seed_opt, extra, outputs, weather_opt])
    subp.call(cmd, shell=True, stdout=subp.DEVNULL)


# -----------------------------
# Helper for replication ignitions (generate Ignitions.csv once)
# -----------------------------
def ensure_replication_ignitions(instance_root, nsims=200, nthreads=4, seed=777):
    """
    If instance_root/Ignitions.csv does not exist and you want weather replication,
    generate it with a small run that logs ignitions and then copy replication.csv
    to Ignitions.csv under the instance.
    """
    target = os.path.join(instance_root, 'Ignitions.csv')
    if os.path.exists(target):
        print(f"[replication] Ignitions.csv already present at {target}")
        return

    tmp_out = os.path.join(instance_root, '_tmp_ignitions')
    os.makedirs(tmp_out, exist_ok=True)

    # Minimal run to write an IgnitionsHistory/replication.csv
    cmd = [
        '/home/ramiro/C2F-W/Cell2Fire/Cell2Fire', '--sim', 'K',
        '--input-instance-folder', instance_root,
        '--output-folder', tmp_out,
        '--nsims', str(nsims),
        '--nthreads', str(nthreads),
        '--seed', str(seed),
        '--cros', '--ignitionsLog',
        '--out-ros',
        '--nweathers', '50', '--weather', 'random',
        '--ignitions-random'
    ]
    subp.call(" ".join(cmd), shell=True, stdout=subp.DEVNULL)

    # Find replication.csv and copy to Ignitions.csv
    repl = glob.glob(os.path.join(tmp_out, 'IgnitionsHistory', 'replication.csv'))
    if repl:
        shutil.copyfile(repl[0], target)
        print(f"[replication] Created {target}")
    else:
        print("[replication] WARNING: replication.csv not found; continuing without replication ignitions.")

    # Clean temp
    try:
        shutil.rmtree(tmp_out)
    except Exception:
        pass


# -----------------------------
# (2) SFB from ROS
# -----------------------------
def surface_fuel_consumed_vectorized(fuels, ros):
    sfc = np.zeros_like(fuels, dtype=np.float32)
    idx1 = (fuels > 0) & (fuels < 6)
    sfc[idx1] = 1 - np.exp(-0.1 * ros[idx1])
    idx2 = (fuels > 5) & (fuels < 14)
    sfc[idx2] = 1 - np.exp(-0.06 * ros[idx2])
    idx3 = (fuels > 13) & (fuels < 30)
    sfc[idx3] = 1 - np.exp(-0.085 * ros[idx3])
    return sfc

def _read_band1(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()

def generate_surface_fraction_burned(ros_asc_path, out_folder, fuels_asc):
    ensure_dir(out_folder)
    fuels, profile = _read_band1(fuels_asc)
    ros, _ = _read_band1(ros_asc_path)
    if fuels.shape != ros.shape:
        raise ValueError(f"Fuels and ROS shapes differ: {fuels.shape} vs {ros.shape}")
    sfb = surface_fuel_consumed_vectorized(fuels, ros)
    out = os.path.join(out_folder, f"SFB_{os.path.basename(ros_asc_path)}")
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(sfb, 1)
    return out

# --- TOP-LEVEL worker (must NOT be nested) ---
def _sfb_worker(args):
    ros_path, fuels_asc, out_folder = args
    return generate_surface_fraction_burned(ros_path, out_folder, fuels_asc)

def build_sfb_from_ros_folder(ros_folder, fuels_asc, out_folder, max_workers=8):
    ensure_dir(out_folder)
    ros_files = [os.path.join(ros_folder, f)
                 for f in os.listdir(ros_folder)
                 if f.lower().endswith(".asc")]
    if not ros_files:
        raise FileNotFoundError(f"No ROS .asc files found in {ros_folder}")
    tasks = [(fp, fuels_asc, out_folder) for fp in ros_files]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_sfb_worker, tasks))
    return out_folder


# -----------------------------
# (3) Burn Probability (from SFB)
# -----------------------------
def bp_from_sfb_folder(sfb_folder, bp_out_asc):
    files = [os.path.join(sfb_folder, f) for f in os.listdir(sfb_folder) if f.lower().endswith(".asc")]
    if not files:
        raise FileNotFoundError(f"No SFB .asc files in {sfb_folder}")
    with rasterio.open(files[0]) as src:
        profile = src.profile.copy()
        count = np.zeros_like(src.read(1), dtype=np.float32)
    for f in files:
        with rasterio.open(f) as src:
            a = src.read(1)
        count += (a > 0).astype(np.float32)
    bp = (count / len(files)).astype(np.float32)
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(bp_out_asc, "w", **profile) as dst:
        dst.write(bp, 1)
    return bp_out_asc


# -----------------------------
# (4) Mean & conditional mean SFB
# -----------------------------
def average_asc_files(input_folder, output_file):
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".asc")]
    if not files:
        raise FileNotFoundError(f"No .asc files in {input_folder}")
    with rasterio.open(files[0]) as src:
        profile = src.profile.copy()
        acc = np.zeros_like(src.read(1), dtype=np.float32)
    for f in files:
        with rasterio.open(f) as src:
            a = src.read(1)
        a[a < 0] = 0
        acc += a
    mean = acc / len(files)
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(mean.astype(np.float32), 1)
    return output_file

def conditional_mean_from_mean_and_bp(mean_sfb_path, bp_path, out_path):
    with rasterio.open(mean_sfb_path) as src:
        mean_sfb = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(bp_path) as src:
        bp = src.read(1)
    cond = np.zeros_like(mean_sfb, dtype=np.float32)
    mask = bp > 0
    cond[mask] = mean_sfb[mask] / bp[mask]
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cond, 1)
    return out_path


# -----------------------------
# (5) Max emissions per pixel
# -----------------------------
def emisiones_maximas(fuels, fuel_load):
    if fuels.shape != fuel_load.shape:
        raise ValueError("fuels and fuel_load must have same shape")
    fuel_consumed = fuel_load
    emisiones = np.zeros_like(fuels, dtype=np.float32)

    idx1 = (fuels > 0) & (fuels < 6)
    eCO2 = 1613 * fuel_consumed[idx1] * 1e-2
    eCH4 = 2.3  * fuel_consumed[idx1] * 1e-2
    eN2O = 0.21 * fuel_consumed[idx1] * 1e-2
    emisiones[idx1] = eCO2 + eCH4 * 27 + eN2O * 273

    idx2 = (fuels > 5) & (fuels < 14)
    eCO2 = 1613 * fuel_consumed[idx2] * 1e-2
    eCH4 = 2.3  * fuel_consumed[idx2] * 1e-2
    eN2O = 0.21 * fuel_consumed[idx2] * 1e-2
    emisiones[idx2] = eCO2 + eCH4 * 27 + eN2O * 273

    idx3 = (fuels > 13) & (fuels < 30)
    eCO2 = 1569 * fuel_consumed[idx3] * 1e-2
    eCH4 = 4.7  * fuel_consumed[idx3] * 1e-2
    eN2O = 0.26 * fuel_consumed[idx3] * 1e-2
    emisiones[idx3] = eCO2 + eCH4 * 27 + eN2O * 273

    return emisiones

def generate_emisiones_maximas_raster(fuels_raster_path, fuel_load_raster, output_raster):
    with rasterio.open(fuels_raster_path) as src:
        fuels = src.read(1); profile = src.profile.copy()
    with rasterio.open(fuel_load_raster) as src:
        fl = src.read(1)
    if fuels.shape != fl.shape:
        raise ValueError("fuels.asc and fuel_load raster differ in shape")
    arr = emisiones_maximas(fuels, fl).astype(np.float32)
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(arr, 1)
    return output_raster


# -----------------------------
# (6) Expected emissions = SFB_cond * Emis_max
# -----------------------------
def multiply_rasters_to_file(a_path, b_path, out_path):
    with rasterio.open(a_path) as A:
        a = A.read(1); profile = A.profile.copy()
    with rasterio.open(b_path) as B:
        b = B.read(1)
    if a.shape != b.shape:
        raise ValueError("rasters must match shape for multiplication")
    c = (a * b).astype(np.float32)
    profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=0, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(c, 1)
    return out_path

def build_expected_emissions_raster(sfb_cond_path, emis_max_path, out_path):
    return multiply_rasters_to_file(sfb_cond_path, emis_max_path, out_path)


# -----------------------------
# (7) DPV from expected emissions
# -----------------------------
def split_into_chunks(data, n_chunks):
    chunk = max(1, len(data) // max(1, n_chunks))
    return [data[i:i + chunk] for i in range(0, len(data), chunk)]

def calculate_value(graphs, values_risk, ncells):
    import networkx as nx
    final = np.zeros(ncells, dtype=np.float32)
    num_graphs = len(graphs)
    for graph in graphs:
        for _, _, attrs in graph.edges(data=True):
            attrs['weight'] = attrs['ros']
        if len(graph.nodes()) == 0:
            continue
        roots = [n for n, d in graph.in_degree() if d == 0]
        if not roots:
            continue
        root = roots[0]
        spaths = nx.single_source_dijkstra_path(graph, root, weight='weight')
        newg = nx.DiGraph()
        for _, path in spaths.items():
            for i in range(len(path)-1):
                newg.add_edge(path[i], path[i+1])
        graph = newg

        nodes = np.array(list(graph.nodes), dtype=np.int32)
        values = np.zeros(ncells, dtype=np.float32)
        desc = {node: np.array(list(nx.descendants(graph, node)), dtype=np.int32) - 1 for node in nodes}
        dpv_vals = np.array([values_risk[d].sum() if len(d) > 0 else 0 for d in desc.values()], dtype=np.float32)
        values[nodes - 1] = values_risk[nodes - 1] + dpv_vals
        final += values / num_graphs
    return final

def process_dpv(graphs, values_risk_file, n_threads, dpv_output):
    header, risk, ncells = read_asc(values_risk_file)
    shape = risk.shape
    flat = risk.reshape([-1])
    blocks = split_into_chunks(graphs, n_threads)
    with ProcessPoolExecutor(max_workers=n_threads) as ex:
        results = ex.map(calculate_value, blocks, [flat]*len(blocks), [ncells]*len(blocks))
    dpv_final = np.zeros(ncells, dtype=np.float32)
    for part in results:
        dpv_final += part
    dpv_final = dpv_final.reshape(shape)
    write_asc(dpv_output, header, dpv_final)
    return dpv_output

def build_dpv_from_messages(messages_folder, risk_raster_path, dpv_out_path, n_threads=8):
    graphs = get_all_messages(messages_folder)
    return process_dpv(graphs, risk_raster_path, n_threads, dpv_out_path)


# -----------------------------
# (8–9) Optimize on DPV & write treatment
# -----------------------------
def model_opt(alpha, dpv_asc, penalty_asc=None, quiet=True):
    """
    Heuristic exact top-k: picks the top-k cells by score.
    score_i = DPV_i - penalty_i (if penalty is provided), else DPV_i.
    Returns (selected_node_ids, header).
    """
    dpv_dic = raster_to_dict(dpv_asc)
    pen_dic = raster_to_dict(penalty_asc) if penalty_asc else {}

    avail = list(dpv_dic.keys())
    k = int(len(avail) * float(alpha))
    if k <= 0:
        header, _, _ = read_asc(dpv_asc)
        return [], header

    scores = {i: (dpv_dic[i] - pen_dic.get(i, 0.0)) for i in avail}
    selected = sorted(scores.keys(), key=lambda i: (scores[i], i), reverse=True)[:k]
    header, _, _ = read_asc(dpv_asc)
    return selected, header

def write_treatment_from_nodes(template_raster_path, node_ids, output_asc_path):
    """
    Create a treatment raster ASC (1 for treated cells, 0 otherwise) using
    node_ids that follow the 'id = 6 + row*ncols + col' convention.
    """
    os.makedirs(os.path.dirname(output_asc_path), exist_ok=True)

    # Read header to get shape and preserve original header text
    with open(template_raster_path, "r") as f:
        header_lines = [next(f) for _ in range(6)]

    meta = {}
    for line in header_lines:
        parts = line.strip().split()
        if not parts:
            continue
        k = parts[0].lower()
        v = parts[1]
        try:
            meta[k] = int(v)
        except ValueError:
            try:
                meta[k] = float(v)
            except ValueError:
                meta[k] = v

    ncols = int(meta["ncols"])
    nrows = int(meta["nrows"])

    data = np.zeros((nrows, ncols), dtype=np.float32)

    for nid in node_ids:
        idx = int(nid) - 6
        if idx < 0:
            continue
        r = idx // ncols
        c = idx % ncols
        if 0 <= r < nrows and 0 <= c < ncols:
            data[r, c] = 1.0

    with open(output_asc_path, "w") as out:
        out.writelines(header_lines)
        np.savetxt(out, data, fmt="%.6f")

    return output_asc_path


# -----------------------------
# (10) Validate with treatment (robust)
# -----------------------------
def c2f_validate(forest, instance_root, out_folder, nsims, nthreads, seed, weather, harvest_csv=None):
    input_folder  = f'--input-instance-folder {instance_root}'
    output_folder = f'--output-folder {out_folder}'
    nsims_opt     = f'--nsims {nsims}'
    nthreads_opt  = f'--nthreads {nthreads}'
    weather_opt   = f'--nweathers 50 --weather {weather}'
    seed_opt      = f'--seed {seed}'

    extra         = '--cros'
    outputs       = '--out-ros --output-messages'

    ignitions_csv = os.path.join(instance_root, 'Ignitions.csv')
    if os.path.exists(ignitions_csv):
        ignitions_flag = '--ignitions'
        print(f"   Using replication ignitions: {ignitions_csv}")
    else:
        ignitions_flag = '--ignitions-random'
        print("   No Ignitions.csv found; using --ignitions-random")

    opts = [input_folder, output_folder, nsims_opt, nthreads_opt, seed_opt, extra, outputs, weather_opt, ignitions_flag]

    if harvest_csv and os.path.exists(harvest_csv) and os.path.getsize(harvest_csv) > 0:
        opts.append(f'--FirebreakCells {harvest_csv}')

    cmd = '/home/ramiro/C2F-W/Cell2Fire/Cell2Fire --sim K ' + " ".join(opts)
    subp.call(cmd, shell=True, stdout=subp.DEVNULL)


# -----------------------------
# Stats & plotting (step 12) + HARVEST EMISSIONS + CI
# -----------------------------
def read_band1(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()

def sum_product(a, b):
    """Return scalar sum of element-wise product of two arrays of same shape."""
    return float((a * b).sum())

def compute_harvest_emissions_sum(emis_max_path, harvest_asc_path):
    """
    Harvest emissions: treated cells emit full per-pixel max emissions.
    """
    emis_max, _ = read_band1(emis_max_path)
    treat, _ = read_band1(harvest_asc_path)
    if emis_max.shape != treat.shape:
        raise ValueError("emis_max and harvest ASC differ in shape")
    return sum_product(emis_max, treat)

def per_sim_wildfire_emissions(emis_max_path, sfb_folder):
    """
    For each SFB file from validation, compute sum(emis_max * SFB_sim).
    Returns a list of totals (one per simulation).
    """
    emis_max, _ = read_band1(emis_max_path)
    files = [os.path.join(sfb_folder, f) for f in os.listdir(sfb_folder) if f.lower().endswith(".asc")]
    totals = []
    for f in files:
        sfb, _ = read_band1(f)
        if sfb.shape != emis_max.shape:
            raise ValueError("SFB and emis_max shapes differ")
        totals.append(sum_product(emis_max, sfb))
    return totals

def mean_sem_ci(values, alpha=0.95):
    """
    Returns (mean, sem, (lo, hi)) using normal approximation.
    """
    vals = np.asarray(values, dtype=np.float64)
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan, (np.nan, np.nan))
    mu = float(vals.mean())
    if n == 1:
        return (mu, np.nan, (mu, mu))
    sd = float(vals.std(ddof=1))
    sem = sd / np.sqrt(n)
    z = 1.959963984540054  # ~N(0,1) 97.5% quantile
    return (mu, sem, (mu - z*sem, mu + z*sem))

def plot_alpha_vs_emissions_ci(alphas, means, ci_lows, ci_highs, out_png):
    plt.figure(figsize=(8, 6))
    plt.fill_between(alphas, ci_lows, ci_highs, color='gray', alpha=0.35, linewidth=0)
    plt.plot(alphas, means, 'o--', color='black', markersize=6, label='Emissions')
    plt.xlabel(r'Percentage of firebreaks ($\alpha$)')
    plt.ylabel(r'CO2E emissions tons ($\varepsilon_\alpha$)')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)


# -----------------------------
# Orchestrator
# -----------------------------
def run_pipeline(cfg):
    f = cfg["forest"]
    P = cfg["paths"]
    S = cfg["sim"]
    O = cfg["opt"]

    # 1) PRESET SIMULATIONS
    print("[1] Running preset simulations…")
    ensure_dir(P["output_preset"])
    c2f_preset(
        forest=f,
        instance_root=P["instance_root"],
        output_preset=P["output_preset"],
        nsims=S["nsims_preset"],
        nthreads=S["nthreads"],
        seed=S["seed_preset"],
        weather=S["weather_preset"]
    )

    # 2) ROS → SFB (preset)
    print("[2] Building SFB from ROS (preset)…")
    build_sfb_from_ros_folder(P["ros_preset"], P["fuels_asc"], P["sfb_preset"])

    # 3) BP from SFB stack (preset)
    print("[3] Computing BP (preset)…")
    bp_from_sfb_folder(P["sfb_preset"], P["bp_preset"])

    # 4) mean SFB and conditional mean given burned (preset)
    print("[4] Mean & conditional SFB (preset)…")
    average_asc_files(P["sfb_preset"], P["mean_sfb_preset"])
    conditional_mean_from_mean_and_bp(P["mean_sfb_preset"], P["bp_preset"], P["sfb_cond_preset"])

    # 5) Emisiones máximas por pixel (preset)
    print("[5] Max emissions (preset)…")
    generate_emisiones_maximas_raster(P["fuels_asc"], P["fl_sup_asc"], P["emis_max_preset"])

    # 6) Expected emissions (preset)
    print("[6] Expected emissions (preset)…")
    build_expected_emissions_raster(P["sfb_cond_preset"], P["emis_max_preset"], P["expected_emis_preset"])

    # 7) DPV from expected emissions (preset)
    print("[7] Building DPV (preset)…")
    build_dpv_from_messages(P["messages_preset"], P["expected_emis_preset"], P["dpv_preset"], n_threads=S["nthreads"])

    # If we want replication in validation, ensure Ignitions.csv exists once
    if S.get("weather_val") == "replication":
        ensure_replication_ignitions(
            instance_root=P["instance_root"],
            nsims=min(200, S.get("nsims_val", 10)),  # small run just to collect points
            nthreads=max(1, S.get("nthreads", 4)),
            seed=S.get("seed_val", 333)
        )

    # 8–12) Loop over treatment alphas
    results_for_plot = []  # (alpha, mean_total, lo, hi)
    alphas = list(map(float, O["alphas"]))

    for alpha in alphas:
        print(f"\n=== Treatment α={alpha} ===")

        # 8) Optimize on DPV
        print("[8] Optimizing on DPV…")
        penalty = P["emis_max_preset"] if O.get("objective","pure") == "dpv_minus_penalty" else None
        selected, header = model_opt(alpha=alpha, dpv_asc=P["dpv_preset"], penalty_asc=penalty)

        # 9) Write treatment rasters + CSV
        print("[9] Writing treatment…")
        ensure_dir(P["harvest_dir"])
        harvest_asc = os.path.join(P["harvest_dir"], f"harvest{alpha}.asc")
        harvest_csv = os.path.join(P["harvest_dir"], f"harvest{alpha}.csv")
        write_treatment_from_nodes(P["fuels_asc"], selected, harvest_asc)
        harvested(harvest_csv, selected)

        # 10) Validate with treatment
        print("[10] Validating with treatment…")
        val_folder = os.path.join(P["validation_root"], str(alpha))
        ensure_dir(val_folder)
        c2f_validate(
            forest=f,
            instance_root=P["instance_root"],
            out_folder=val_folder,
            nsims=S["nsims_val"],
            nthreads=S["nthreads"],
            seed=S["seed_val"],
            weather=S["weather_val"],
            harvest_csv=harvest_csv
        )

        # 11) Repeat steps 2–6 on validation outputs
        print("[11] Postprocessing validation outputs…")
        ros_val = os.path.join(val_folder, "RateOfSpread")
        if not os.path.isdir(ros_val) or not os.listdir(ros_val):
            print(f"WARNING: Missing validation outputs at {ros_val}. Skipping alpha={alpha}.")
            continue

        sfb_val = os.path.join(val_folder, "SurfFractionBurn_selected")
        mean_sfb_val = os.path.join(val_folder, "mean_sfb.asc")
        bp_val = os.path.join(val_folder, "bp.asc")
        sfb_cond_val = os.path.join(val_folder, "mean_sfb_cond.asc")
        expected_emis_val = os.path.join(val_folder, "expected_emis.asc")

        build_sfb_from_ros_folder(ros_val, P["fuels_asc"], sfb_val)
        bp_from_sfb_folder(sfb_val, bp_val)
        average_asc_files(sfb_val, mean_sfb_val)
        conditional_mean_from_mean_and_bp(mean_sfb_val, bp_val, sfb_cond_val)
        build_expected_emissions_raster(sfb_cond_val, P["emis_max_preset"], expected_emis_val)

        # 12) Stats with HARVEST EMISSIONS and CI
        print("[12] Summarizing… (includes harvest emissions)")
        try:
            # constant harvest emissions for this alpha
            harvest_total = compute_harvest_emissions_sum(P["emis_max_preset"], harvest_asc)

            # per-sim wildfire totals
            wildfire_totals = per_sim_wildfire_emissions(P["emis_max_preset"], sfb_val)

            # combine per-sim totals with harvest constant
            total_per_sim = [wt + harvest_total for wt in wildfire_totals]

            mean_total, sem_total, (lo, hi) = mean_sem_ci(total_per_sim)
            results_for_plot.append((alpha, mean_total, lo, hi))

            print(f"   Harvest emissions: {harvest_total:,.2f} tCO2e")
            print(f"   Wildfire mean:     {np.mean(wildfire_totals):,.2f} tCO2e")
            print(f"   TOTAL mean:        {mean_total:,.2f} tCO2e")
            print(f"   95% CI:            [{lo:,.2f}, {hi:,.2f}]")
        except Exception as e:
            print(f"   ERROR computing stats for alpha={alpha}: {e}")

    # Plot (only for alphas that produced results)
    out_plot = os.path.join(P["results_root"], "expected_emissions_vs_alpha_ci.png")
    ensure_dir(P["results_root"])

    if not results_for_plot:
        print("\nNo validation outputs were generated for any alpha; skipping plot.")
    else:
        a_vals, m_vals, lo_vals, hi_vals = zip(*results_for_plot)
        plot_alpha_vs_emissions_ci(list(a_vals), list(m_vals), list(lo_vals), list(hi_vals), out_plot)
        print(f"\nSaved plot with CI: {out_plot}")
    print("Done.")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Raster post-processing pipeline")
    ap.add_argument("--forest", type=str, default="heterogeneo", help="Forest name")
    ap.add_argument("--alphas", type=str, default="0.0,0.03,0.05,0.06,0.07", help="Comma-separated alphas")
    ap.add_argument("--nsims-preset", type=int, default=10)
    ap.add_argument("--nsims-val", type=int, default=10)
    ap.add_argument("--objective", type=str, default="dpv_minus_penalty",
                    choices=["pure", "dpv_minus_penalty"])
    # Optional overrides
    ap.add_argument("--fuels-asc", type=str, default=None)
    ap.add_argument("--fl-sup-asc", type=str, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = default_config(args.forest)
    cfg["opt"]["alphas"] = [float(a) for a in args.alphas.split(",")]
    cfg["sim"]["nsims_preset"] = args.nsims_preset
    cfg["sim"]["nsims_val"] = args.nsims_val
    cfg["opt"]["objective"] = args.objective

    if args.fuels_asc: cfg["paths"]["fuels_asc"] = args.fuels_asc
    if args.fl_sup_asc: cfg["paths"]["fl_sup_asc"] = args.fl_sup_asc

    # sanity
    for must in ["fuels_asc", "fl_sup_asc"]:
        if not os.path.exists(cfg["paths"][must]):
            sys.exit(f"ERROR: Missing required raster: {cfg['paths'][must]}")

    run_pipeline(cfg)







