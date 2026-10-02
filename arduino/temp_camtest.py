import serial
import time

PORT = '/dev/ttyUSB0'
BAUD = 115200
OUTPUT_FILE = 'test.jpg'

def capture_photo():
    print(f"Öppnar {PORT}...")
    ser = serial.Serial()
    ser.port = PORT
    ser.baudrate = BAUD
    ser.timeout = 5

    # Förhindra att baskortet fastnar i reset-läge
    ser.dtr = False
    ser.rts = False
    ser.open()

    # Tryck "mjuk reset" via RTS för att tvinga fram en ren omstart
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False

    print("Väntar på att kameran ska starta upp...")
    time.sleep(2)

    # Läs ut eventuell boot-text från kameran
    while ser.in_waiting:
        boot_msg = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"[CAM LOG]: {boot_msg}")

    print("Skickar CAPTURE...")
    ser.write(b"CAPTURE\n")

    # Läs svaret
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    print(f"Svar från kamera: '{line}'")

    if line.startswith("SIZE:"):
        img_size = int(line.split(":")[1])
        print(f"Läser in bild ({img_size} bytes)...")

        img_data = ser.read(img_size)

        if len(img_data) == img_size:
            with open(OUTPUT_FILE, 'wb') as f:
                f.write(img_data)
            print(f"Klart! Bild sparad som '{OUTPUT_FILE}'.")
        else:
            print(f"Fel: Fick {len(img_data)} av {img_size} bytes.")

    ser.close()

if __name__ == '__main__':
    capture_photo()
