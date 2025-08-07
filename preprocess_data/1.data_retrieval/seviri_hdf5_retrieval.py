import argparse
import fnmatch
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import eumdac
import eumdac.config
import eumdac.tailor_models
import eumdac.customisation

# import h5py
import numpy as np
import requests
import urllib3
import xarray as xr
from pathos.threading import ThreadPool as Pool
from pyresample import create_area_def
from satpy.scene import Scene

OUTPUTDIR = Path("/data/WeatherGenerator/SEVIRI")
CREDENTIALS_FILE = eumdac.config.get_credentials_path()


def get_credentials():
    with CREDENTIALS_FILE.open("r") as f:
        content = f.read()
    return eumdac.token.Credentials(*content.strip().split(","))


ALL_CHANNELS = [
    "channel_1",
    "channel_2",
    "channel_3",
    "channel_4",
    "channel_5",
    "channel_6",
    "channel_7",
    "channel_8",
    "channel_9",
    "channel_10",
    "channel_11",
]


def import_SEVIRI(file_path: str):
    scn = Scene(reader="seviri_l1b_native", filenames=[file_path])
    scn.load(
        [
            "IR_016",
            "IR_039",
            "IR_087",
            "IR_097",
            "IR_108",
            "IR_120",
            "IR_134",
            "VIS006",
            "VIS008",
            "WV_062",
            "WV_073",
        ]
    )
    return scn

def generate_area_def(
    scene: Scene, min_lon: float, max_lon: float, min_lat: float, max_lat: float
):
    proj_dict = {"proj": "longlat", "datum": "WGS84"}

    # Calculate the resolution from the original scene's area extent and shape
    orig_area = scene.finest_area()  # Get the finest area from the scene
    orig_shape = (
        orig_area.width,
        orig_area.height,
    )  # Original width and height in pixels

    # Calculate the resolution in degrees/pixel
    lons, lats = scene["IR_108"].attrs["area"].get_lonlats()
    lons[lons==np.inf] = np.nan
    lats[lats==np.inf] = np.nan
    lon_res = (np.nanmin(lons) - np.nanmax(lons)) / orig_shape[0]
    lat_res = (np.nanmin(lats) - np.nanmax(lats)) / orig_shape[1]

    return create_area_def(
        "my_area",
        proj_dict,
        area_extent=(min_lon, min_lat, max_lon, max_lat),
        units="degrees",
        resolution=(lon_res, lat_res),
    )


def regrid_reproject(
    scene: Scene, min_lon: float, max_lon: float, min_lat: float, max_lat: float
) -> Scene:
    proj_dict = {"proj": "longlat", "datum": "WGS84"}

    # Calculate the resolution from the original scene's area extent and shape
    orig_area = scene.finest_area()  # Get the finest area from the scene
    orig_shape = (
        orig_area.width,
        orig_area.height,
    )  # Original width and height in pixels

    # Calculate the resolution in degrees/pixel
    lons, lats = scene["IR_108"].attrs["area"].get_lonlats()
    lons[lons==np.inf] = np.nan
    lats[lats==np.inf] = np.nan
    lon_res = (np.nanmin(lons) - np.nanmax(lons)) / orig_shape[0]
    lat_res = (np.nanmin(lats) - np.nanmax(lats)) / orig_shape[1]

    new_area = create_area_def(
        "my_area",
        proj_dict,
        area_extent=(min_lon, min_lat, max_lon, max_lat),
        units="degrees",
        resolution=(lon_res, lat_res),
    )
    return scene.resample(new_area, mode="nearest", retain_values=True)


def get_collection(collection, start_time, end_time, credentials):
    token = eumdac.token.AccessToken(credentials)
    datastore = eumdac.datastore.DataStore(token)

    selected_collection = datastore.get_collection(collection)
    print(f"{selected_collection} - {selected_collection.title}")

    # Retrieve datasets that match our filter
    products = selected_collection.search(dtstart=start_time, dtend=end_time)
    print(f"Found Datasets: {products.total_results} datasets for the given time range")
    return products


def download_api_products(
    products,
    output_dir: str,
    datatailor: eumdac.datatailor.DataTailor,
    chain: eumdac.tailor_models.Chain,
):
    """Download and regrid SEVIRI data.

    The native files are downloaded and regridded and reprojected using the Satpy
    library.
    Note: the Native files aren't deleted.
    """
    sleep_time = 5

    for product in products:
        year = str(product).split("-")[5][0:4]
        month = str(product).split("-")[5][4:6]
        output_direc = os.path.join(output_dir, year, month)
        os.makedirs(output_direc, exist_ok=True)

        file = Path(output_direc) / f"{product}.nc"
        if not file.exists():
            # try:
            customisation = datatailor.new_customisation(product, chain)
            stream = None
            while True:
                try:
                    status = customisation.status
                except eumdac.customisation.UnableToGetCustomisationError:
                    print("Unable to get customization. Retrying")
                    time.sleep(30)
                    status = customisation.status
                
                if "DONE" in status:
                    try:
                        output = customisation.outputs
                    except eumdac.customisation.UnableToGetCustomisationError:
                        time.sleep(10)
                        output = customisation.outputs
                    zip_files = fnmatch.filter(output, "*")[0]

                    with customisation.stream_output(zip_files) as stream:
                        fname = OUTPUTDIR / stream.name
                        # Check if stream.name (the file path) already exists
                        if not fname.exists():
                            # If the file doesn't exist, open it for writing
                            try:
                                with fname.open(mode="wb") as fdst:
                                    shutil.copyfileobj(stream, fdst)
                            # Retry upon connection error
                            except urllib3.exceptions.ProtocolError:
                                time.sleep(10)
                                with fname.open(mode="wb") as fdst:
                                    shutil.copyfileobj(stream, fdst)
                            print(f"File '{stream.name}' created and saved.")
                        else:
                            print(
                                f"File '{stream.name}' already exists. Skipping creation."
                            )

                    print(
                        f"Download finished for customisation {customisation._id}."
                    )
                    break
                elif status in ["ERROR", "FAILED", "DELETED", "KILLED", "INACTIVE"]:
                    print(
                        f"Customisation {customisation._id} was unsuccessful. Log is printed.\n"
                    )
                    print(customisation.logfile)
                    try:
                        customisation.delete()
                    except eumdac.customisation.CustomisationError as error:
                        print("Customisation Error:", error)
                    except requests.exceptions.RequestException as error:
                        print("Unexpected error:", str(error))
                    break
                elif "QUEUED" in status:
                    print(f"Customisation {customisation._id} is queued.")
                time.sleep(sleep_time)

            if stream is not None:
                file_path = OUTPUTDIR / stream.name
                scn = import_SEVIRI(str(file_path))
                print("file imported")

                try:
                    customisation.delete()
                except eumdac.customisation.CustomisationError:
                    print("failed to delete customization.")

                # reproject the file in right format
                rpj_scn = regrid_reproject(scn, min_lon, max_lon, min_lat, max_lat)
                print("file reprojected")

                # Define the output path for the HDF5 file
                output_path = f"{output_direc}/{product}.nc"

                # Save to HDF5 using the NetCDF4 engine
                rpj_scn.save_datasets(filename=output_path, engine="netcdf4")

                with xr.load_dataset(output_path, engine="netcdf4") as ds:
                    ds = ds.drop_vars(["longitude", "latitude"])
                    comp = dict(zlib=True, complevel=9) # compress data
                    encoding = {var: comp for var in ds.data_vars}    
                    ds.to_netcdf(f"{output_direc}/{product}.nc", encoding=encoding)
                os.remove(file_path)  # remove .nat file
            # except:
            #     print("unexpected error occured")
            #     for customisation in datatailor.customisations:
            #         if customisation.status in ["INACTIVE"]:
            #             customisation.kill()
            #             try:
            #                 customisation.delete()
            #             except eumdac.customisation.CustomisationError as error:
            #                 print("Customisation Error:", error)
            #             except Exception as error:
            #                 print("Unexpected error:", error)

            #             print(
            #                 f"Delete {customisation.status} customisation {customisation} from {customisation.creation_time} UTC."
            #             )

            #         elif customisation.status in [
            #             "ERROR",
            #             "FAILED",
            #             "DELETED",
            #             "KILLED",
            #         ]:
            #             try:
            #                 customisation.delete()
            #             except eumdac.customisation.CustomisationError as error:
            #                 print("Customisation Error:", error)
            #             except requests.exceptions.RequestException as error:
            #                 print("Unexpected error:", error)

            #             print(
            #                 f"Delete completed customisation {customisation} from {customisation.creation_time} UTC."
            #             )


def parallel_download_api_products(
    list_of_products,
    list_of_dirs: list[str],
    datatailor: eumdac.datatailor.DataTailor,
    chain: eumdac.tailor_models.Chain,
    threads=4,
):
    # Set number of threads (cores) used for parallel run and map threads
    if threads is None:
        pool = Pool()
    else:
        pool = Pool(nodes=threads)
    datatailors = [datatailor] * len(list_of_dirs)
    chains = [chain] * len(list_of_dirs)
    results = pool.map(download_api_products, list_of_products, list_of_dirs, datatailors, chains)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="seviri-cli",
        description="Download seviri data from EUMDAC",
    )
    parser.add_argument("year")
    parser.add_argument("month")
    args = parser.parse_args()
    year = int(args.year)
    month = int(args.month)
    # Insert your personal key and secret into the single quotes
    credentials = get_credentials()

    print("Requesting token.")
    token = eumdac.token.AccessToken(credentials)

    print("Accessing datastore.")
    datastore = eumdac.datastore.DataStore(token)
    datatailor = eumdac.datatailor.DataTailor(token)

    # Define collection
    collection = "EO:EUM:DAT:MSG:HRSEVIRI"

    # Set sensing start and end time
    start = datetime(year, month, 1, 0, 0)
    if month < 12:
        end = datetime(year, month+1, 1, 0, 0)
    else:
        end = datetime(year+1, 1, 1, 0, 0)

    # Bounding box (in degrees)
    min_lon = 3
    max_lon = 8
    min_lat = 50
    max_lat = 54

    print("Retrieving collection:")
    selected_collection = datastore.get_collection(collection)
    print(f"    {selected_collection} - {selected_collection.title}")

    products = selected_collection.search(dtstart=start, dtend=end)
    print(f"Found Datasets: {products.total_results} datasets for the given time range")

    datatailor = eumdac.datatailor.DataTailor(token)

    # To check if Data Tailor works as expected, we are requesting our quota information
    print("DataTailor quota: ", datatailor.quota)

    chain = eumdac.tailor_models.Chain(
        product="HRSEVIRI",
        format="msgnative",
        filter={"bands": ALL_CHANNELS},
        roi={"NSWE": [max_lat, min_lat, min_lon, max_lon]},
    )
    prod_list = list(products)
    product = prod_list[0]
    customisation = datatailor.new_customisation(product, chain)

    customisation.status

    # Create nested list of products for parallel pool
    nested_products = [[x] for x in products]
    list_of_dirs = [str(OUTPUTDIR)] * len(nested_products)

    # Parallel processing with timing
    start = time.time()
    # parallel_download_api_products(nested_products, list_of_dirs, datatailor, chain)
    download_api_products(products, str(OUTPUTDIR), datatailor, chain)
    stop = time.time()
    print(f"Execution time (minutes): {(stop - start) / 60}")

    # # Sometimes, because of multiple failed download files, your workspace exceeds
    # # the maximum number of 25 GB. With this code you can clean your online workspace,
    # #  to make room for new download requests.
    # for customisation in datatailor.customisations:
    #     if customisation.status in ["QUEUED", "INACTIVE", "RUNNING"]:
    #         customisation.kill()
    #         print(
    #             f"Delete {customisation.status} customisation {customisation} from {customisation.creation_time} UTC."
    #         )
    #         try:
    #             customisation.delete()
    #         except eumdac.datatailor.CustomisationError as error:
    #             print("Customisation Error:", error)
    #         except Exception as error:
    #             print("Unexpected error:", error)
    #     else:
    #         print(
    #             f"Delete completed customisation {customisation} from {customisation.creation_time} UTC."
    #         )
    #         try:
    #             customisation.delete()
    #         except eumdac.datatailor.CustomisationError as error:
    #             print("Customisation Error:", error)
    #         except requests.exceptions.RequestException as error:
    #             print("Unexpected error:", error)
