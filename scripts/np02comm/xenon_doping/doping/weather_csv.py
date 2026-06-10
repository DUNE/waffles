from bme280 import BME280
from smbus2 import SMBus
from time import sleep
from datetime import datetime
import csv
import os

# Sensore
bus = SMBus(1)
sensor = BME280(i2c_dev=bus)

# Scarta prima lettura
sensor.get_temperature()
sensor.get_pressure()
sensor.get_humidity()
sleep(1)

file_csv = "/home/simone/weather.csv"

# Se file non esiste ? crea intestazione
file_esiste = os.path.exists(file_csv)

with open(file_csv, mode='a', newline='') as file:
    writer = csv.writer(file)

    if not file_esiste:
        writer.writerow(["Date", "Time", "Temp (C)", "Pressure (hPa)", "Humidity (%)"])

    try:
        while True:
            now = datetime.now()

            data = now.strftime("%Y-%m-%d")
            ora = now.strftime("%H:%M:%S")
            temperatura = round(sensor.get_temperature(), 1)
            pressione = round(sensor.get_pressure(), 1)
            umidita = round(sensor.get_humidity(), 1)

            print(data, ora, temperatura, pressione, umidita)

            writer.writerow([data, ora, temperatura, pressione, umidita])
            file.flush()   # salva subito su disco

            sleep(1)   

    except KeyboardInterrupt:
        print("\nProgramma terminato")
