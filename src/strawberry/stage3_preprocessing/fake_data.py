from sensor_ocr import generate_sensor_ocr_data


def generate_fake_env_data():
    print("fake_data.py is deprecated; using Sensor OCR for temperature_c and humidity_pct.")
    generate_sensor_ocr_data()


if __name__ == "__main__":
    generate_fake_env_data()
