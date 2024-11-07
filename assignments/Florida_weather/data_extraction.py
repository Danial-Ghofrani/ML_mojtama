### Flood prediction in Miami
import mysql.connector
import requests
import requests_cache
import pandas as pd
import _mysql_connector
from mysql.connector import Error


requests_cache.install_cache("open_meteo_cache", expire_after = -1)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 25.7617,
    "longitude": -80.1918,
    "start_date": "2000-01-01",
    "end_date": "2024-11-06",
    "hourly": ["temperature_2m", "relative_humidity_2m", "daw_point_2m", "apparent_temperature",
               "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl",
               "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
               "wind_speed_10m", "wind_speed_100m", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
               "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm"],
    "timezone": "America/New_York",
    "temporal_resolution": "hourly_6"
}

response = requests.get(url, params=params)
data = response.json()

if "hourly" in data:
    data_range = pd.data_range(start = params["start_date"], end = params["end_date"], freq = "6H")
    hourly_data = {"data": data_range}
    for var_name, values in data["hourly"].items():
        if len(values) == len(data_range):
            hourly_data[var_name] = values

        else:
            print(f" Warning: Length mismatch for {var_name}")
    if all(len(values) == len(data_range) for values in hourly_data.values()):
        hourly_df = pd.DataFrame(hourly_data)


        try:
            connection = mysql.connector.connect(
                host = "",
                user = "",
                password = "",
                database = ""
            )