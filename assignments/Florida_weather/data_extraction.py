### Flood prediction in Miami
import requests
import pandas as pd




class Weather:

    def __init__(self, city):
        self.city = city

    def get_openmeteo_data(self, url, params):


        url = url
        params = params
        # Make the API request
        response = requests.get(url, params=params)

        # Check if the response is successful
        if response.status_code == 200:
            data = response.json()

            # Extract hourly data
            hourly_data = data.get("hourly", {})

            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(hourly_data)

            # Save DataFrame to CSV
            csv_filename = f"{self.city}_weather_data.csv"
            df.to_csv(csv_filename, index=False)

            print(f"Data successfully saved to {csv_filename}")
        else:
            print("Error:", response.status_code)


    def load_and_compute_thresholds(self):

        df = pd.read_csv(f"{self.city}_weather_data.csv")

        self.precipitation_threshold = df["precipitation"].quantile(0.95)
        self.soil_moisture_0_to_7cm_threshold = df["soil_moisture_0_to_7cm"].quantile(0.95)
        self.soil_moisture_7_to_28cm_threshold = df["soil_moisture_7_to_28cm"].quantile(0.95)
        self.soil_moisture_28_to_100cm_threshold = df["soil_moisture_28_to_100cm"].quantile(0.95)
        self.soil_moisture_100_to_255cm_threshold = df["soil_moisture_100_to_255cm"].quantile(0.95)
        self.snow_depth_threshold = df["snow_depth"].quantile(0.95)



    def is_flood(self, row):


        if (row["precipitation"] > self.precipitation_threshold or
            row["soil_moisture_0_to_7cm"] > self.soil_moisture_0_to_7cm_threshold or
            row["soil_moisture_7_to_28cm"] > self.soil_moisture_7_to_28cm_threshold or
            row["soil_moisture_28_to_100cm"] > self.soil_moisture_28_to_100cm_threshold or
            row["soil_moisture_100_to_255cm"] > self.soil_moisture_100_to_255cm_threshold or
            row["snow_depth"] > self.snow_depth_threshold):
            return 1

        else:

            return 0


    def flood_column(self):

        df = pd.read_csv(f"{self.city}_weather_data.csv")
        df["flood"] = df.apply(self.is_flood, axis=1)
        df.drop(columns="time")


        df.to_csv("final_flood_data.csv", index=False)
        print("Flood prediction column added successfully!")


# url = "https://archive-api.open-meteo.com/v1/archive"
# params = {
#     "latitude": 25.7617,  # Miami, Florida
#     "longitude": -80.1918,
#     "start_date": "2000-01-01",
#     "end_date": "2024-11-06",
#     "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
#               "precipitation,rain,snowfall,snow_depth,weather_code,pressure_msl,"
#               "surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
#               "wind_speed_10m,wind_speed_100m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,"
#               "soil_moisture_28_to_100cm,soil_moisture_100_to_255cm",
#     "timezone": "America/New_York"
# }
#
mi_Weather = Weather("miami")
# # mi_Weather.get_openmeteo_data(url=url, params=params)

# mi_Weather.load_and_compute_thresholds()
# mi_Weather.flood_column()

data = pd.read_csv("final_flood_data.csv")
print(data[data["flood"] == 0])


