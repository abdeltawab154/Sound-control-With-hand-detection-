import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Volume Control")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        vol_range = self.volume.GetVolumeRange()
        self.min_vol = vol_range[0]
        self.max_vol = vol_range[1]

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        hand_control = False
        current_volume_level = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                thumb_coords = np.array([thumb_tip.x * frame.shape[1], thumb_tip.y * frame.shape[0]], dtype=int)
                index_coords = np.array([index_tip.x * frame.shape[1], index_tip.y * frame.shape[0]], dtype=int)
                
                distance = np.linalg.norm(thumb_coords - index_coords)
                volume_level = np.interp(distance, [20, 200], [self.min_vol, self.max_vol])
                self.volume.SetMasterVolumeLevel(volume_level, None)
                hand_control = True
                
                # Draw green line indicator between thumb and index finger
                cv2.line(frame, tuple(thumb_coords), tuple(index_coords), (0, 255, 0), 3)
                
                # Calculate and store current volume level
                current_volume_level = np.interp(volume_level, [self.min_vol, self.max_vol], [0, 100])

        if current_volume_level is not None:
            # Display current volume level on the frame
            cv2.putText(frame, f'Volume: {int(current_volume_level)}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
