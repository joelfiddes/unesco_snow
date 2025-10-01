import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

# === Configuration ===
myvar = "Rof"
scenarios = {
    "ssp126": "/home/joel/sim/cmip6_daily_data/quantileMap/outputs/fsm_ssp126.nc",
    "ssp245": "/home/joel/sim/cmip6_daily_data/quantileMap/outputs/fsm_ssp245.nc",
    "ssp585": "/home/joel/sim/cmip6_daily_data/quantileMap/outputs/fsm_ssp585.nc",
}
hist_file = "/home/joel/sim/cmip6_daily_data/quantileMap/outputs/fsm_hist.nc"
centroids_file = "/home/joel/sim/cmip6_daily_data/quantileMap/outputs/df_centroids.pck"
basins_shp = "/home/joel/Data/CA_basins/basins.shp"

output_file = "/home/joel/sim/cmip6_daily_data/unesco_report_code/code_clean/casnow_app/data/processed_"+myvar+".csv"

# === Helper Functions ===

def load_and_mask_data(ds, var):
    mask_limits = {"SWE": 1000, "snd": 2, "alb": 1000, "Tsf": 1000, "Rof": 1000}
    arr = ds[var]
    limit = mask_limits.get(var)
    if limit is not None:
        arr = arr.where(arr <= limit, np.nan)
    if var == "alb":
        arr = arr.where(arr >= 0, np.nan)
    return arr

def assign_hydro_day_of_year(da):
    doy = da.time.dt.dayofyear
    hydro_doy = np.where(doy >= 244, doy - 244, doy + 121)
    da.coords["hydrological_day_of_year"] = ("time", hydro_doy)
    return da

def load_spatial_data(centroids_path, basins_path):
    df_centroids = pd.read_pickle(centroids_path)
    basins = gpd.read_file(basins_path)
    sample_points = gpd.GeoDataFrame(
        df_centroids,
        geometry=gpd.points_from_xy(df_centroids["lon"], df_centroids["lat"]),
        crs=basins.crs
    )
    joined = gpd.sjoin(sample_points, basins[["REGION", "geometry"]], how="inner", op="intersects")
    return df_centroids, joined.groupby("REGION")

def extract_elevation_bins(df_centroids):
    elevation_bins = np.arange(0, np.max(df_centroids["elevation"]) + 1000, 1000)
    bin_indices = np.digitize(df_centroids["elevation"], elevation_bins) - 1
    return elevation_bins, bin_indices

# === Run Analysis ===

df_centroids, regions = load_spatial_data(centroids_file, basins_shp)
elevation_bins, elevation_bin_indices = extract_elevation_bins(df_centroids)
elevation_labels = np.arange(len(elevation_bins) - 1)

results = []

# Historical
ds_hist = xr.open_dataset(hist_file)
swehist = assign_hydro_day_of_year(load_and_mask_data(ds_hist, myvar))
time_hist = (ds_hist.time.dt.year >= 1981) & (ds_hist.time.dt.year <= 2010)
swehist = swehist.sel(time=time_hist)

for scenario_name, path in scenarios.items():
    ds = xr.open_dataset(path)
    swe_future = assign_hydro_day_of_year(load_and_mask_data(ds, myvar))
    time_future = (ds.time.dt.year >= 2081) & (ds.time.dt.year <= 2100)
    swe_future = swe_future.sel(time=time_future)

    for region, region_data in regions:
        sample_indices = region_data.index

        for elev_bin_id in elevation_labels:
            elev_range = (elevation_bins[elev_bin_id], elevation_bins[elev_bin_id + 1])
            samples_bin = sample_indices[elevation_bin_indices[sample_indices] == elev_bin_id]
            if len(samples_bin) == 0:
                continue

            # Historical
            hist_bin = swehist.isel(sample=samples_bin)
            hist_mean = hist_bin.groupby("hydrological_day_of_year").mean(dim=["time", "sample"])
            hist_mean = hist_mean.rolling(hydrological_day_of_year=10, center=True).mean()

            # Future
            fut_bin = swe_future.isel(sample=samples_bin)
            fut_mean = fut_bin.groupby("hydrological_day_of_year").mean(dim=["time", "sample"])
            fut_mean = fut_mean.rolling(hydrological_day_of_year=10, center=True).mean()

            for day in range(1, 366):
                results.append({
                    "scenario": scenario_name,
                    "region": region,
                    "elev_low": elev_range[0],
                    "elev_high": elev_range[1],
                    "hydro_day": day,
                    "historical_mean": hist_mean.sel(hydrological_day_of_year=day, method="nearest").values.item(),
                    "future_mean": fut_mean.sel(hydrological_day_of_year=day, method="nearest").values.item()
                })

# === Save Result ===
df_result = pd.DataFrame(results)
df_result.to_csv(output_file)
print(f"Saved processed data to {output_file}")
