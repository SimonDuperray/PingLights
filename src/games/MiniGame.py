import serial
import time
import random
import pygame


SERIAL_PORT = "COM3"
BAUD_RATE = 9600
GAME_DURATION = 60
NUM_ZONES = 6


class MiniGame:
    def __init__(self):
        pygame.mixer.init()

        # chargement des sons
        self.sound_go = pygame.mixer.Sound("assets/sounds/mk_beep.wav")
        self.sound_success = pygame.mixer.Sound("assets/sounds/success.wav")
        self.sound_fail = pygame.mixer.Sound("assets/sounds/fail.wav")

        # variables du jeu
        self.score = 0
        self.current_zone = None
        self.game_active = False
        self.start_time = None
        try:
            self.arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            print("Arduino connecté")
        except:
            self.arduino = None
            print("Arduino non connecté, mode simulation")

    def send_zone(self, zone):
        if self.arduino:
            self.arduino.write(f"ZONE:{zone}\n".encode())
        print(f"Zone allumée : {zone}")

    def turn_off_all(self):
        if self.arduino:
            for i in range(1, NUM_ZONES+1):
                self.arduino.write(f"ZONE:{i+1}\n".encode())
        print("LEDs éteintes.")

    def pick_next_zone(self):
        zones = list(range(1, NUM_ZONES+1))
        if self.current_zone in zones:
            zones.remove(self.current_zone)
        self.current_zone = random.choice(zones)
        return self.current_zone

    def get_time_left(self):
        elapsed = time.time() - self.start_time
        return max(0, GAME_DURATION - elapsed)

    def start(self):
        print("==== DEBUT DU JEU ====")
        self.score = 0
        self.game_active = False

        self.sound_go.play()
        time.sleep(4.5)

        self.game_active = True
        self.start_time = time.time()
        # Allume la première zone
        zone = self.pick_next_zone()
        self.send_zone(zone)

        return zone

    def on_bounce_detected(self, detected_zone):
        if not self.game_active:
            return

        time_left = self.get_time_left()

        if time_left <= 0:
            self.end_game()
            return

        if detected_zone == self.current_zone:
            self.score += 1
            self.sound_success.play()
            print(f"Bonne zone ! Score : {self.score} | Temps restant : {time_left:.1f}")
            zone = self.pick_next_zone()
            self.send_zone(zone)
        else:
            self.sound_fail.play()
            print(f"Mauvaise zone (détectée : {detected_zone}, attendue : {self.current_zone})")

    def end_game(self):
        self.game_active = False
        self.turn_off_all()
        print(f"==== FIN DU JEU ==== Score final: {self.score}")

    def run(self):
        self.start()
        while self.game_active:
            if self.get_time_left() <= 0:
                self.end_game()
            time.sleep(0.1)

if __name__ == "__main__":
    game = MiniGame()
    game.run()