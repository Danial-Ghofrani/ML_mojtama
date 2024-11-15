### Flood prediction in Miami
import requests
import pandas as pd

# Define the URL and parameters for the OpenMeteo Archive API request
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 25.7617,  # Miami, Florida
    "longitude": -80.1918,
    "start_date": "2000-01-01",
    "end_date": "2024-11-06",
    "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
              "precipitation,rain,snowfall,snow_depth,weather_code,pressure_msl,"
              "surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
              "wind_speed_10m,wind_speed_100m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,"
              "soil_moisture_28_to_100cm,soil_moisture_100_to_255cm",
    "timezone": "America/New_York"
}

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
    csv_filename = "miami_weather_data.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Data successfully saved to {csv_filename}")
else:
    print("Error:", response.status_code)







# if "hourly" in data:
#     date_range = pd.date_range(start = params["start_date"], end = params["end_date"], freq = "6H")
#     hourly_data = {"data": date_range}
#     for var_name, values in data["hourly"].items():
#         if len(values) == len(date_range):
#             hourly_data[var_name] = values
#
#         else:
#             print(f" Warning: Length mismatch for {var_name}")
#     if all(len(values) == len(date_range) for values in hourly_data.values()):
#         hourly_df = pd.DataFrame(hourly_data)
#
#         try:
#             connection = mysql.connector.connect(
#                 host="localhost",
#                 user="root",
#                 password="root123",
#                 database="Miami_weather"
#             )
#
#             if connection.is_connected():
#                 cursor = connection.cursor()
#
#                 # Corrected table creation query
#                 table_creation_query = """
#                 CREATE TABLE IF NOT EXISTS florida_weather_data (
#                     date DATETIME,
#                     temperature_2m FLOAT,
#                     relative_humidity_2m FLOAT,
#                     dew_point_2m FLOAT,
#                     apparent_temperature FLOAT,
#                     precipitation FLOAT,
#                     rain FLOAT,
#                     snowfall FLOAT,
#                     snow_depth FLOAT,
#                     weather_code INT,
#                     pressure_msl FLOAT,
#                     surface_pressure FLOAT,
#                     cloud_cover FLOAT,
#                     cloud_cover_low FLOAT,
#                     cloud_cover_mid FLOAT,
#                     cloud_cover_high FLOAT,
#                     wind_speed_10m FLOAT,
#                     wind_speed_100m FLOAT,
#                     soil_moisture_0_to_7cm FLOAT,
#                     soil_moisture_7_to_28cm FLOAT,
#                     soil_moisture_28_to_100cm FLOAT,
#                     soil_moisture_100_to_255cm FLOAT
#                 );
#                 """
#                 cursor.execute(table_creation_query)
#
#                 # Corrected data insertion loop
#                 for i, row in hourly_df.iterrows():
#                     insert_query = """
#                     INSERT INTO florida_weather_data (
#                         date, temperature_2m, relative_humidity_2m, dew_point_2m, apparent_temperature,
#                         precipitation, rain, snowfall, snow_depth, weather_code, pressure_msl,
#                         surface_pressure, cloud_cover, cloud_cover_low, cloud_cover_mid, cloud_cover_high,
#                         wind_speed_10m, wind_speed_100m, soil_moisture_0_to_7cm, soil_moisture_7_to_28cm,
#                         soil_moisture_28_to_100cm, soil_moisture_100_to_255cm
#                     ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#                     """
#                     # Convert row to tuple and insert
#                     data_tuple = tuple(row)
#                     cursor.execute(insert_query, data_tuple)
#
#                 # Commit and confirm
#                 connection.commit()
#                 print("Data inserted into MySQL database successfully.")
#
#         except Error as e:
#             print(f"Error: {e}")
#
#         finally:
#             if connection.is_connected():
#                 cursor.close()
#                 connection.close()
#                 print("MySQL connection is closed")
#
#     else:
#         print("Error: not all variables have the same length as the date range. Data not inserted.")
# else:
#     print("Error: Expected hourly data structured not found in the response.")


# Path to your SQLite file
# sqlite_file = "open_meteo_cache.sqlite"
#
# # Connect to SQLite database
# conn = sqlite3.connect(sqlite_file)
#
# # Fetch all table names from SQLite
# cursor = conn.cursor()
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = cursor.fetchall()
#
# # Export each table to a CSV file
# for table_name in tables:
#     table_name = table_name[0]  # Table name is a tuple, so extract the name
#     df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
#     df.to_csv(f"{table_name}.csv", index=False)  # Export table to CSV
#     print(f"Exported {table_name} to {table_name}.csv")
#
# # Close the connection
# conn.close()