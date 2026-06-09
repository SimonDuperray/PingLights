import serial
import time


def send_on(ser):
    ser.write(b"ON\n")


def send_off(ser):
    ser.write(b"OFF\n")

if __name__ == "__main__":
    ser = serial.Serial("COM5", 9600, timeout=1)
    time.sleep(2)

    send_on(ser)
    print("Message envoyé -> LEDS ALLUMEES")
    response = ser.readline().decode("utf-8").strip()
    print(f"Arduino dit : {response}")

    time.sleep(5)

    send_off(ser)
    print("Message envoyé -> LEDS ETEINTES")
    response = ser.readline().decode("utf-8").strip()
    print(f"Arduino dit : {response}")

    ser.close()
