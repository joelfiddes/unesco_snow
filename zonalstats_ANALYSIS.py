import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
import rasterio
import os
import json

# Paths & files setup (same as before)
tif_files = [

    "/media/joel/LaCie/data/gef_snow/outputs/rof_ssp126_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/rof_ssp245_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/rof_ssp585_2081-01-01_2100-12-31.tif",

    "/media/joel/LaCie/data/gef_snow/outputs/snd_ssp126_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/snd_ssp245_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/snd_ssp585_2081-01-01_2100-12-31.tif",


    "/media/joel/LaCie/data/gef_snow/outputs/swe_ssp126_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/swe_ssp245_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/swe_ssp585_2081-01-01_2100-12-31.tif",

    "/media/joel/LaCie/data/gef_snow/outputs/gst_ssp126_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/gst_ssp245_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/gst_ssp585_2081-01-01_2100-12-31.tif",

    "/media/joel/LaCie/data/gef_snow/outputs/alb_ssp126_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/alb_ssp245_2081-01-01_2100-12-31.tif",
    "/media/joel/LaCie/data/gef_snow/outputs/alb_ssp585_2081-01-01_2100-12-31.tif"

]

# Historical files
hist_files = {


    "swe": "/media/joel/LaCie/data/gef_snow/outputs/swe_hist_1981-01-01_2010-01-01.tif",
    "snd": "/media/joel/LaCie/data/gef_snow/outputs/snd_hist_1981-01-01_2010-01-01.tif",
    "rof": "/media/joel/LaCie/data/gef_snow/outputs/rof_hist_1981-01-01_2010-01-01.tif",
    "gst": "/media/joel/LaCie/data/gef_snow/outputs/gst_hist_1981-01-01_2010-01-01.tif",
    "alb": "/media/joel/LaCie/data/gef_snow/outputs/alb_hist_1981-01-01_2010-01-01.tif",
}

boundary_shapefile = "./casnow_app/data/basins.shp"
basins = gpd.read_file(boundary_shapefile)

variables = ["gst", "rof", "swe", "snd", "alb"]
scenarios = ["ssp126", "ssp245", "ssp585"]

output_dir = "./data/"
os.makedirs(output_dir, exist_ok=True)

for variable in variables:
    for scenario in scenarios:
        future_files = [f for f in tif_files if variable in f and scenario in f]
        if not future_files:
            print(f"No future file found for {variable} {scenario}")
            continue

        future_tif = future_files[0]
        hist_tif = hist_files.get(variable)
        if hist_tif is None:
            print(f"No historical file found for {variable}")
            continue

        with rasterio.open(future_tif) as src_future, rasterio.open(hist_tif) as src_hist:
            future_data = src_future.read(1).astype(np.float32)
            hist_data = src_hist.read(1).astype(np.float32)
            raster_crs = src_hist.crs

            if future_data.shape != hist_data.shape:
                print(f"Shape mismatch for {variable}: {future_data.shape} vs {hist_data.shape}")
                continue

            if basins.crs != raster_crs:
                basins = basins.to_crs(raster_crs)

            anomaly = future_data - hist_data
            anomaly = np.ma.masked_invalid(anomaly)

            meta = src_future.meta.copy()
            meta.update(dtype=rasterio.float32)

            anomaly_tif = os.path.join(output_dir, f"{variable}_{scenario}_anomaly.tif")
            with rasterio.open(anomaly_tif, "w", **meta) as dest:
                dest.write(anomaly.filled(np.nan), 1)

        stats = zonal_stats(
            basins,
            anomaly_tif,
            stats=["mean"],
            geojson_out=True
        )

        if not stats:
            print(f"No zonal stats produced for {variable} {scenario}")
            continue

        # Save stats as GeoJSON (could also save as Shapefile or GeoPackage)
        stats_gdf = gpd.GeoDataFrame.from_features(stats)
        stats_gdf.set_crs(raster_crs, inplace=True)   # assign CRS before reprojection
        stats_gdf = stats_gdf.to_crs(epsg=4326)


        output_stats_path = os.path.join(output_dir, f"{variable}_{scenario}_zonal_stats.geojson")
        stats_gdf.to_file(output_stats_path, driver="GeoJSON")

        print(f"Saved zonal stats for {variable} {scenario} to {output_stats_path}")





