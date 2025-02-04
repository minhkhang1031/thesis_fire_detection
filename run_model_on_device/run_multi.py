import subprocess

yolo_process = subprocess.Popen(["python3", "run_camera.py"])

sensor_process = subprocess.Popen(["python3", "run_sensor.py"])

yolo_process.wait()
sensor_process.wait()
