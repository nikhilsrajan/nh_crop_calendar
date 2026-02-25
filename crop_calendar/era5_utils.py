import os
import datetime
import time
import xarray as xr
import rioxarray as rio


def load_mean_temperature_nc_file(nc_filepath:str)->xr.DataArray:
    ncfile = xr.open_dataset(nc_filepath)
    t2m = ncfile['t2m']
    t2m = t2m - 273.15 # K to deg C
    t2m.coords['longitude'] = (t2m.coords['longitude'] + 180) % 360 - 180
    t2m = t2m.sortby('longitude')
    t2m = t2m.rio.set_spatial_dims('longitude', 'latitude')
    t2m = t2m.rio.write_crs('epsg:4326')
    return t2m


def load_total_precipitation_nc_file(nc_filepath:str):
    ncfile = xr.open_dataset(nc_filepath)
    tp = ncfile['tp']
    tp = tp * 1000 # m to mm
    tp.coords['longitude'] = (tp.coords['longitude'] + 180) % 360 - 180
    tp = tp.sortby('longitude')
    tp = tp.rio.set_spatial_dims('longitude', 'latitude')
    tp = tp.rio.write_crs('epsg:4326')
    return tp
