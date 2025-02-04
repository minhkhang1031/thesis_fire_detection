import adafruit_dht
import RPi.GPIO as GPIO
import time
import board
import subprocess
import threading

dht_device = adafruit_dht.DHT22(board.D27)
MQ2_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(MQ2_PIN, GPIO.IN)

alert_lock = threading.Lock()
def play_alert_sound():
    with alert_lock:
        for _ in range(1):
            subprocess.call(['mpg321', "alert_sound.mp3"])
            time.sleep(10)
            
def check_fire_conditions():
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        mq2_status = GPIO.input(MQ2_PIN)

        if mq2_status == 0 or temperature > 54:
            threading.Thread(target=play_alert_sound, daemon=True).start()
        else:
            print("MQ-2: Normal.")  
            print(f"Temperature: {temperature:.1f}C")

    except RuntimeError as error:   
        print(f"Reading error: {error}")

try:
    while True:
        check_fire_conditions()
        time.sleep(2)
except KeyboardInterrupt:
    print("Program stopped.")
finally:
    GPIO.cleanup()
