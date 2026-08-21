"""
laptop_subscriber.py

Listens to the MQTT topic "alpr/plate" on the local Mosquitto broker and
prints every plate the Raspberry Pi sends. Run on the laptop:
    py laptop_subscriber.py
"""

import json
import logging
import signal
import sys

import paho.mqtt.client as mqtt


# ---- Config: where the broker lives and which topic we listen on ----
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
TOPIC       = "alpr/plate"
CLIENT_ID   = "laptop-subscriber"


# ---- Logging: nice timestamped output on the terminal ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sub")


# ---- MQTT callbacks: react to connect / disconnect / incoming plate ----
def on_connect(client, userdata, flags, *args):
    rc = args[0] if args else 0
    if rc == 0:
        log.info("Connected to %s:%s, subscribing to %s", BROKER_HOST, BROKER_PORT, TOPIC)
        client.subscribe(TOPIC, qos=1)
    else:
        log.error("Connect failed, rc=%s", rc)


def on_disconnect(client, userdata, *args):
    log.warning("Disconnected, will auto-reconnect...")


def on_message(client, userdata, msg):
    raw = msg.payload.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
        plate = data.get("plate", "?")
        conf = data.get("confidence", 0)
        log.info("PLATE: %-12s  conf=%.2f", plate, float(conf))
    except json.JSONDecodeError:
        log.info("PLATE: %s", raw)


# ---- Main: build the client, hook callbacks, connect, listen forever ----
def main():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    except AttributeError:
        client = mqtt.Client(client_id=CLIENT_ID)

    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Clean exit on Ctrl+C.
    def handle_sig(*_):
        log.info("Stopping...")
        client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()
