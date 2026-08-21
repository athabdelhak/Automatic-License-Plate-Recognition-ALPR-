<img width="964" height="1280" alt="photo_2026-08-21_03-29-37" src="https://github.com/user-attachments/assets/b2be0a37-ccad-4d38-8cc5-9af454c88ed1" />
# 🚗 Edge-Based Automatic License Plate Recognition (ALPR)

## Project Overview
This project implements a real-time, edge-computing Automatic License Plate Recognition (ALPR) system. Designed to run on a Raspberry Pi 5 with a camera module, it utilizes a dual-stage vision pipeline to accurately detect, track, and extract license plate text from live video feeds. To ensure high accuracy, the system incorporates temporal stability tracking and majority-voting logic before publishing structured JSON data to a remote client over MQTT.

## Key Features
* **Real-Time Edge Inference:** Runs detection and OCR locally on the Raspberry Pi 5.
* **Dual-Stage Vision Pipeline:** Uses custom YOLOv8 for plate localization and PaddleOCR for robust alphanumeric text recognition.
* **Temporal Stability Tracking:** Leverages Intersection over Union (IoU) to verify that a detected vehicle is stationary/stable across consecutive frames before running heavy OCR.
* **Majority-Voting & Confidence Filtering:** Buffers multiple OCR reads to find a consensus, drastically reducing false readings and OCR artifacts.
* **Smart Noise Rejection:** Built-in rules discard automaker logos (e.g., BMW, Audi), website URLs, and invalid character sequences.
* **IoT Event-Driven Architecture:** Publishes lightweight JSON payloads (license plate string, confidence score, timestamp) via MQTT to a central monitoring client.

## Tech Stack
* **Language:** Python 3.x
* **Computer Vision & AI:** OpenCV, Ultralytics YOLOv8, PaddleOCR
* **Hardware Interface:** Picamera2
* **IoT & Messaging:** Paho-MQTT, Eclipse Mosquitto

## Note 
* The project was supposed to be submitted in May, but a bit of laziness took over and some extra features were added. It was also presented as a project at ENSTI Algeria by a group of three people, and I am the person responsible for writing the code and connecting it with all the necessary hardware.
* Go to tag v1.0 to install yolov8 model (Best.pt)
* The file "_debug.py" is for program testing in static pictures ,I provided some images to test it .
* Use AI to understand the code and how to run it better. Don't forget to follow me on LinkedIn at this link: https://www.linkedin.com/in/abdelhak-athamena-927212319/


## SETUP :
## 📂 Repository Structure
* **`rpi_alpr.py`**: Main edge pipeline running on the Raspberry Pi 5.
* **`laptop_subscriber.py`**: MQTT consumer script running on your laptop.
* **`_debug.py`**: Offline debugging script for tuning OCR thresholds.
* **`best.pt`**: Your custom YOLOv8 model weights for license plate detection.

## 🛠️ Prerequisites
* **Hardware:** Raspberry Pi 5 with Pi Camera v1.3, and a PC/Laptop.
* **Network:** Both devices must be on the same local network (LAN/Wi-Fi).
* **MQTT Broker:** Eclipse Mosquitto installed on the laptop.

## ⚙️ Setup & Installation

### 1. Laptop (MQTT Broker & Subscriber)
1. **Install Mosquitto Broker:**
   * **Windows/Mac:** Download and install from [mosquitto.org](https://mosquitto.org/download/).
   * **Linux (Ubuntu/Debian):** `sudo apt install mosquitto mosquitto-clients`
2. **Start the Mosquitto service** (if it doesn't start automatically).
3. **Install Python dependencies:**
   ```bash
   pip install paho-mqtt
   ```

### 2. Raspberry Pi 5 (Edge Detection)
1. Ensure your Pi Camera is connected and configured.
2. Install the necessary computer vision and IoT libraries:
   ```bash
   pip install ultralytics paddleocr paddlepaddle paho-mqtt opencv-python
   ```
*(Note: `picamera2` is usually pre-installed on recent Raspberry Pi OS versions).*

## 🚀 How to Launch the Project

### Step 1: Configure the Network IP
1. Find your laptop's local IPv4 address (e.g., `10.44.152.219`).
2. Open `rpi_alpr.py` on your Raspberry Pi.
3. Update the `BROKER_HOST` variable to match your laptop's IP:
   ```python
   BROKER_HOST = "10.44.152.219"  # <--- Change this to your laptop's IP
   ```

### Step 2: Start the Laptop Subscriber
Open a terminal on your laptop and run the listener script. It will connect to the local broker and wait for plates:
```bash
python laptop_subscriber.py
```

### Step 3: Start the Edge Pipeline
Open a terminal on your Raspberry Pi and launch the main pipeline:
```bash
python rpi_alpr.py
```

**What happens next?**
1. The Pi camera will warm up.
2. A live preview window will open (if a display is connected) showing the YOLO bounding boxes.
3. When a plate is stable for 3 frames, PaddleOCR reads it.
4. If the plate passes the majority-vote logic, the Pi sends a JSON payload to the laptop.
5. You will see the plate number and confidence score print out in your laptop's terminal!
