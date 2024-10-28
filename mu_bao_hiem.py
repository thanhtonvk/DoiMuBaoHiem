import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Toplevel
import cv2
import threading
import time
import re
import numpy as np
from ultralytics import YOLO
import pygame
import requests
from PIL import Image, ImageTk
from scipy.spatial import distance
import uuid
from datetime import datetime
from modules.helmet_detection import predictHelmet
from modules.bicycle_detection import combineBoxes

# Initialize pygame for sound playback
pygame.init()
pygame.mixer.init()


def is_intersect(x_min1, y_min1, x_max1, y_max1, x_min2, y_min2, x_max2, y_max2):
    if x_max1 < x_min2 or x_max2 < x_min1:
        return False
    if y_max1 < y_min2 or y_max2 < y_min1:
        return False
    return True


frame_count = 0
start_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX

font_scale = 1  # Kích thước font chữ
color = (0, 255, 0)  # Màu chữ (xanh lá cây)
thickness = 2
# Telegram bot configuration
TELEGRAM_BOT_TOKEN = "7233650823:AAGr1Cmpr56o4NBdJFyloFUDfltqjnA1dwA"
TELEGRAM_CHAT_ID = "7131930827"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
TELEGRAM_PHOTO_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
classes = ["khong doi mu", "doi mu"]


def send_telegram_message(message):
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(TELEGRAM_API_URL, data=data)


def send_telegram_photo(frame):
    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {"photo": ("image.jpg", img_encoded.tobytes())}
    data = {"chat_id": TELEGRAM_CHAT_ID}
    requests.post(TELEGRAM_PHOTO_URL, data=data, files=files)


class CameraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Interface")
        self.geometry("1280x720")

        self.TRANG_THAI = ""
        self.is_normal_state = False  # Biến kiểm soát trạng thái "BÌNH THƯỜNG"
        # Load and set the background image
        self.background_image = Image.open("hugo.jpg")
        self.background_image = self.background_image.resize(
            (self.winfo_screenwidth(), self.winfo_screenheight()),
            Image.Resampling.LANCZOS,
        )
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(self, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Add logo at the top left
        self.logo_image = Image.open("logo.png")
        self.logo_image = self.logo_image.resize((100, 100), Image.Resampling.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        self.logo_label = tk.Label(self, image=self.logo_photo, bg="#2c3e50")
        self.logo_label.place(x=20, y=10)

        # Create a title frame with an inner red border
        self.title_frame = tk.Frame(self, bg="#00BFFF", bd=10, relief="ridge")
        self.title_frame.place(x=200, y=2, width=1000, height=90)

        self.title_label = tk.Label(
            self.title_frame,
            text="HỆ THỐNG PHÁT HIỆN VÀ CẢNH BÁO HỌC SINH ĐỘI MŨ BẢO HIỂM",
            font=("Helvetica", 18, "bold"),
            fg="#FF0000",
        )
        self.title_label.pack(padx=20, pady=20)

        # Khởi tạo đồng hồ số
        self.clock_label = tk.Label(
            self, font=("Helvetica", 16), fg="#FF0000", bg="#F5F5F5"
        )
        self.clock_label.place(x=25, y=20)
        # Khởi tạo các đối tượng cho nhận diện khuôn mặt và nhận diện đối tượng
        self.video_source_left = 0

        self.cap_left = None

        self.running_left = False
        self.lock_left = threading.Lock()
        self.modeYolo = "END"

        self.font_large = ("Helvetica", 16, "bold")
        self.font_small = ("Helvetica", 12)

        button_width = 15
        button_height = 2

        # Camera frame 1 (Chỉnh sửa kích thước khung hình và canvas)
        self.frame_camera1 = tk.Frame(
            self,
            width=600,
            height=480,
            bg="#00BFFF",
            bd=10,
            relief="solid",
            highlightbackground="#00BFFF",
            highlightcolor="#00BFFF",
            highlightthickness=4,
        )
        self.frame_camera1.place(x=360, y=120)

        # Canvas for Camera 1 (Cập nhật kích thước canvas)
        self.canvas_left = tk.Canvas(
            self.frame_camera1, width=600, height=480, bg="#34495e"
        )
        self.canvas_left.pack()

        # Buttons frame 1 (Sắp xếp lại vị trí các nút)
        self.frame_buttons1 = tk.Frame(self, bg="#2c3e50")
        self.frame_buttons1.place(x=360, y=650)  # Căn chỉnh lại vị trí các nút

        self.button_camera1 = tk.Button(
            self.frame_buttons1,
            text="Camera",
            font=self.font_small,
            bg="#1abc9c",
            fg="#ffffff",
            width=button_width,
            height=button_height,
            bd=2,
            relief="solid",
            highlightbackground="#16a085",
            command=self.start_camera_left,
        )
        self.button_camera1.pack(side=tk.LEFT, padx=10)

        self.button_video1 = tk.Button(
            self.frame_buttons1,
            text="Video",
            font=self.font_small,
            bg="#1abc9c",
            fg="#ffffff",
            width=button_width,
            height=button_height,
            bd=2,
            relief="solid",
            highlightbackground="#16a085",
            command=self.load_video_left,
        )
        self.button_video1.pack(side=tk.LEFT, padx=10)
        self.button_stop = tk.Button(
            self.frame_buttons1,
            text="Dừng âm thanh",
            font=self.font_small,
            bg="#1abc9c",
            fg="#ffffff",
            width=button_width,
            height=button_height,
            bd=2,
            relief="solid",
            highlightbackground="#16a085",
            command=self.stopAmThanh,
        )
        self.button_stop.pack(side=tk.LEFT, padx=10)
        self.button_exit1 = tk.Button(
            self.frame_buttons1,
            text="Exit",
            font=self.font_small,
            bg="#1abc9c",
            fg="#ffffff",
            width=button_width,
            height=button_height,
            bd=2,
            relief="solid",
            highlightbackground="#16a085",
            command=self.quit,
        )
        self.button_exit1.pack(side=tk.LEFT, padx=10)

        self.add_hover_effects(
            [self.button_camera1, self.button_video1, self.button_exit1]
        )

        self.add_hover_effects(
            [self.button_camera1, self.button_video1, self.button_exit1]
        )

        self.fps_start_time_left = time.time()
        self.fps_start_time_right = time.time()
        self.frame_count_left = 0
        self.frame_count_right = 0

        self.points_left = []
        self.polygons_left = []
        self.is_play_audio = True

    def play_doi_mu(self):
        print("hoc sinh khong doi mu")
        if not pygame.mixer.music.get_busy() and self.is_play_audio:
            pygame.mixer.music.load("Alarm/alarm.wav")
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
            pygame.time.set_timer(pygame.USEREVENT, 30000)

    def stopAmThanh(self):
        pygame.mixer.music.stop()
        self.is_play_audio = False

    def startCheckin(self):
        self.mode = "START_CHECKIN"

    def endCheckin(self):
        self.mode = "NONE"

    def startCheckout(self):
        self.mode = "START_CHECKOUT"

    def endCheckout(self):
        self.mode = "NONE"

    def add_hover_effects(self, buttons):
        for button in buttons:
            button.bind("<Enter>", self.on_enter)
            button.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        event.widget.config(fg="#2980b9")

    def on_leave(self, event):
        event.widget.config(fg="#ffffff")

    def start_camera_left(self):
        if not self.running_left:
            self.running_left = True
            threading.Thread(target=self.process_left, daemon=True).start()

    def load_video_left(self):
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
        )
        if video_path:
            if self.running_left:
                self.running_left = False
                time.sleep(1)  # Wait for the previous thread to close
            threading.Thread(
                target=self.process_video_left, args=(video_path,), daemon=True
            ).start()

    def connect_camera_left(self):
        pass

    def predictMuBaoHiem(self, frame):
        self.result = predictHelmet(frame)

    def process_left(self):
        count_frame = 0
        self.cap_left = cv2.VideoCapture(self.video_source_left)
        while self.running_left:
            with self.lock_left:
                ret, frame = self.cap_left.read()
            if ret:
                count_frame += 1
                resultBike = combineBoxes(frame)
                if resultBike is not None:
                    print(resultBike)
                    boxes, labels = resultBike
                    for box, label in zip(boxes, labels):
                        x_min, y_min, x_max, y_max = box
                        cropped = frame[y_min:y_max, x_min:x_max]
                        resultHelmet = predictHelmet(cropped)
                        if resultHelmet:
                            boxesHelmet, labelsHelmet = resultHelmet
                            for labelHelmet in labelsHelmet:
                                if labelHelmet == 0:
                                    threading.Thread(target=self.play_doi_mu).start()
                                    threading.Thread(
                                        target=send_telegram_message,
                                        args=("Học sinh không đội mũ",),
                                    ).start()
                                    threading.Thread(
                                        target=send_telegram_photo, args=(frame,)
                                    ).start()

                                cv2.putText(
                                    frame,
                                    classes[labelHelmet],
                                    (x_min, y_min),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 0, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                        cv2.rectangle(
                            frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
                        )

                self.display_frame_thread_safe(
                    frame, self.canvas_left, self.points_left, self.polygons_left
                )
                self.frame_count_left += 1
                elapsed_time = time.time() - self.fps_start_time_left
                if elapsed_time > 1:
                    fps = self.frame_count_left / elapsed_time
                    self.fps_start_time_left = time.time()
                    self.frame_count_left = 0
                    self.update_fps_display(self.canvas_left, round(fps))
            else:
                break
        with self.lock_left:
            self.cap_left.release()

    def process_video_left(self, video_path):
        self.cap_left = cv2.VideoCapture(video_path)
        count_frame = 0
        while self.cap_left.isOpened():
            with self.lock_left:
                ret, frame = self.cap_left.read()
            if ret:
                count_frame += 1
                resultBike = combineBoxes(frame)
                if resultBike is not None:
                    print(resultBike)
                    boxes, labels = resultBike
                    for box, label in zip(boxes, labels):
                        x_min, y_min, x_max, y_max = box
                        xmin1,ymin1,xmax1,ymax1 = box
                        cropped = frame[y_min:y_max, x_min:x_max]
                        resultHelmet = predictHelmet(cropped)
                        if resultHelmet:
                            boxesHelmet, labelsHelmet = resultHelmet
                            for boxHelmet,labelHelmet in zip(boxesHelmet,labelsHelmet):
                                if labelHelmet == 0:
                                    # threading.Thread(target=self.play_doi_mu).start()
                                    threading.Thread(
                                        target=send_telegram_message,
                                        args=("Học sinh không đội mũ",),
                                    ).start()
                                    threading.Thread(
                                        target=send_telegram_photo, args=(frame,)
                                    ).start()
                                xmin2,ymin2,xmax2,ymax2 = boxHelmet
                                w = xmax2-xmin2
                                h = ymax2-ymin2
                                xmin3 = xmin1+xmin2
                                xmax3 = xmin3+w
                                ymin3 = ymin1+ymin2
                                ymax3 = ymin3+h
                                cv2.rectangle(frame, (xmin3, ymin3), (xmax3, ymax3), (255, 255, 0), 2)
                                
                                
                                cv2.putText(
                                    frame,
                                    classes[labelHelmet],
                                    (x_min, y_min),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 0, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                        cv2.rectangle(
                            frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
                        )
                self.display_frame_thread_safe(
                    frame, self.canvas_left, self.points_left, self.polygons_left
                )
                self.frame_count_left += 1
                elapsed_time = time.time() - self.fps_start_time_left
                if elapsed_time > 1:
                    fps = self.frame_count_left / elapsed_time
                    self.fps_start_time_left = time.time()
                    self.frame_count_left = 0
                    self.update_fps_display(self.canvas_left, round(fps))
            else:
                break
        with self.lock_left:
            self.cap_left.release()

    def display_frame_thread_safe(self, frame, canvas, points, polygons):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 600))
        img = tk.PhotoImage(master=canvas, data=cv2.imencode(".png", img)[1].tobytes())
        canvas.after(0, self.update_canvas, canvas, img, points, polygons)

    def update_canvas(self, canvas, img, points, polygons):
        canvas.create_image(0, 0, image=img, anchor=tk.NW)
        canvas.img = img
        for point in points:
            x, y = point
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="green")
        for polygon in polygons:
            canvas.create_polygon(polygon, outline="blue", fill="", width=2)

    def update_fps_display(self, canvas, fps):
        canvas.delete("fps")
        canvas.create_text(
            10,
            10,
            anchor=tk.NW,
            text=f"FPS: {fps}",
            fill="red",
            font=self.font_small,
            tag="fps",
        )

    def quit(self):
        self.running_left = False
        self.running_right = False
        if self.cap_left is not None:
            with self.lock_left:
                self.cap_left.release()
        # if self.cap_right is not None:
        #     with self.lock_right:
        #         self.cap_right.release()
        cv2.destroyAllWindows()
        self.destroy()


if __name__ == "__main__":
    app = CameraApp()

    def handle_pygame_events():
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:
                pygame.mixer.music.stop()
                pygame.time.set_timer(pygame.USEREVENT, 0)

    def pygame_loop():
        while True:
            handle_pygame_events()
            time.sleep(0.1)

    threading.Thread(target=pygame_loop, daemon=True).start()
    app.mainloop()
