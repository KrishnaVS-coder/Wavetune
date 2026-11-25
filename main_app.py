import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import numpy as np
import time
import pyautogui
import os
import sys

# Global variables
model = None
classes = ["A", "B", "C", "D", "E"]
prediction_history = []
last_action_time = 0
last_executed_sign = None
action_delay = 1.5
running = False
current_sign = "None"
cap = None
no_detection_count = 0
confidence_threshold = 0.7

def load_model_delayed():
    """Load model after UI is shown"""
    global model
    try:
        import tensorflow as tf
        # Works both as script and as .exe
        if getattr(sys, 'frozen', False):
            BASE_DIR = os.path.dirname(sys.executable)
        else:
            BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        
        MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model.h5")
        print(f"Looking for model at: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model not found at:\n{MODEL_PATH}\n\nCopy models folder next to WaveTune.exe")
            return
        
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        messagebox.showinfo("Ready", "WaveTune is ready!\n\nClick START to begin.")
    except Exception as e:
        print(f"Model load error: {e}")
        messagebox.showerror("Error", f"Model error:\n{str(e)}")

def start_control():
    global running, prediction_history, last_executed_sign, no_detection_count
    
    if model is None:
        messagebox.showerror("Error", "Model not loaded!\n\nRestart the application.")
        return
    
    prediction_history = []
    last_executed_sign = None
    no_detection_count = 0
    running = True
    start_btn.config(state="disabled", bg="#555555")
    stop_btn.config(state="normal", bg="#E74C3C")
    status_label.config(text="🟢 Active", fg="#00ff00")
    
    threading.Thread(target=run_detection, daemon=True).start()

def stop_control():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    cv2.destroyAllWindows()
    
    start_btn.config(state="normal", bg="#27AE60")
    stop_btn.config(state="disabled", bg="#555555")
    status_label.config(text="🔴 Stopped", fg="#E74C3C")
    sign_label.config(text="---")
    action_label.config(text="None")

def run_detection():
    global last_action_time, current_sign, cap, last_executed_sign, no_detection_count
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera!")
        stop_control()
        return
    
    # Hand detection setup
    hand_cascade = None
    try:
        hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
    except:
        pass

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Simple hand detection - check center region
        h, w = frame.shape[:2]
        roi_frame = frame[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)]
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Check if there's significant content (hand)
        _, thresh = cv2.threshold(gray_roi, 60, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.size
        hand_ratio = white_pixels / total_pixels
        
        # If too uniform (no hand), skip prediction
        if hand_ratio < 0.15 or hand_ratio > 0.98:
            no_detection_count += 1
            if no_detection_count > 3:
                sign_label.config(text="No Hand")
                status_label.config(text="🟡 Waiting", fg="#FFA500")
                last_executed_sign = None
                prediction_history.clear()
                current_sign = "None"
            cv2.putText(frame, "No Hand Detected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_control()
                break
            continue
        
        img = cv2.resize(frame, (64, 64))
        img = img.astype("float32") / 255.0
        img = np.reshape(img, (1, 64, 64, 3))

        try:
            prediction = model.predict(img, verbose=0)
            class_index = np.argmax(prediction)
            confidence = prediction[0][class_index]
            
            # High confidence threshold
            if confidence < 0.90:
                no_detection_count += 1
                if no_detection_count > 5:
                    sign_label.config(text="No Hand")
                    status_label.config(text="🟡 Waiting", fg="#FFA500")
                    last_executed_sign = None
                    prediction_history.clear()
                continue
            
            no_detection_count = 0
            status_label.config(text="🟢 Active", fg="#00ff00")
            
            prediction_history.append(class_index)
            if len(prediction_history) > 7:
                prediction_history.pop(0)

            if len(prediction_history) >= 5:
                final_index = max(set(prediction_history), key=prediction_history.count)
                label = classes[final_index]
                current_sign = label
                sign_label.config(text=label)

                # Execute only if different from last executed sign
                if label != last_executed_sign:
                    if label == "A":
                        pyautogui.press("playpause")
                        action_label.config(text="⏯ Play/Pause")
                        last_executed_sign = label
                        print(f"Executed: A (Play/Pause)")
                    elif label == "B":
                        pyautogui.press("nexttrack")
                        action_label.config(text="⏭ Next Track")
                        last_executed_sign = label
                        print(f"Executed: B (Next Track)")
                    elif label == "C":
                        pyautogui.press("prevtrack")
                        action_label.config(text="⏮ Previous Track")
                        last_executed_sign = label
                        print(f"Executed: C (Previous Track)")
                    elif label == "D":
                        pyautogui.press("volumeup")
                        action_label.config(text="🔊 Volume Up")
                        last_executed_sign = label
                        print(f"Executed: D (Volume Up)")
                    elif label == "E":
                        pyautogui.press("volumedown")
                        action_label.config(text="🔉 Volume Down")
                        last_executed_sign = label
                        print(f"Executed: E (Volume Down)")
                    
                    prediction_history.clear()
        except Exception as e:
            print(f"Error: {e}")

        cv2.putText(frame, f"Sign: {current_sign}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_control()
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

def on_closing():
    if running:
        stop_control()
    root.destroy()
    sys.exit()

# Create UI
root = tk.Tk()
root.title("WaveTune - Gesture Control")
root.geometry("550x650")
root.configure(bg="#2C3E50")
root.resizable(True, True)
root.protocol("WM_DELETE_WINDOW", on_closing)

# Header
header = tk.Frame(root, bg="#34495E", height=80)
header.pack(fill="x")

title = tk.Label(header, text="🎵 WaveTune", 
                font=("Arial", 20, "bold"), bg="#34495E", fg="#ECF0F1")
title.pack(pady=10)

subtitle = tk.Label(header, text="Gesture-Controlled Music Player", 
                   font=("Arial", 10), bg="#34495E", fg="#BDC3C7")
subtitle.pack()

# Main Content
content = tk.Frame(root, bg="#2C3E50")
content.pack(fill="both", expand=True, padx=20, pady=10)

# Status Panel
status_frame = tk.Frame(content, bg="#34495E", relief="raised", bd=2)
status_frame.pack(fill="x", pady=5)

tk.Label(status_frame, text="Status:", font=("Arial", 12, "bold"), 
         bg="#34495E", fg="#ECF0F1").pack(side="left", padx=15, pady=10)
status_label = tk.Label(status_frame, text="🔴 Stopped", 
                       font=("Arial", 12, "bold"), bg="#34495E", fg="#E74C3C")
status_label.pack(side="left", pady=10)

# Detection Panel
detect_frame = tk.Frame(content, bg="#34495E", relief="raised", bd=2)
detect_frame.pack(fill="x", pady=5)

tk.Label(detect_frame, text="Sign:", font=("Arial", 12, "bold"), 
         bg="#34495E", fg="#ECF0F1").pack(side="left", padx=15, pady=10)
sign_label = tk.Label(detect_frame, text="---", 
                     font=("Arial", 22, "bold"), bg="#34495E", fg="#3498DB")
sign_label.pack(side="left", pady=10, padx=10)

# Action Panel
action_frame = tk.Frame(content, bg="#34495E", relief="raised", bd=2)
action_frame.pack(fill="x", pady=5)

tk.Label(action_frame, text="Action:", font=("Arial", 12, "bold"), 
         bg="#34495E", fg="#ECF0F1").pack(side="left", padx=15, pady=10)
action_label = tk.Label(action_frame, text="None", 
                       font=("Arial", 12), bg="#34495E", fg="#2ECC71")
action_label.pack(side="left", pady=10)

# Guide Panel
guide_frame = tk.Frame(content, bg="#34495E", relief="raised", bd=2)
guide_frame.pack(fill="both", expand=True, pady=5)

tk.Label(guide_frame, text="📋 Commands", 
         font=("Arial", 13, "bold"), bg="#34495E", fg="#ECF0F1").pack(pady=8)

gestures = [
    ("Sign A", "⏯ Play/Pause"),
    ("Sign B", "⏭ Next Track"),
    ("Sign C", "⏮ Previous"),
    ("Sign D", "🔊 Volume Up"),
    ("Sign E", "🔉 Volume Down")
]

for sign, action in gestures:
    row = tk.Frame(guide_frame, bg="#34495E")
    row.pack(fill="x", padx=20, pady=4)
    
    tk.Label(row, text=sign, font=("Arial", 11, "bold"), 
             bg="#34495E", fg="#3498DB", width=10, anchor="w").pack(side="left")
    tk.Label(row, text="→", font=("Arial", 12, "bold"), 
             bg="#34495E", fg="#E67E22").pack(side="left", padx=10)
    tk.Label(row, text=action, font=("Arial", 11), 
             bg="#34495E", fg="#ECF0F1", anchor="w").pack(side="left")

# Control Buttons
btn_frame = tk.Frame(content, bg="#2C3E50")
btn_frame.pack(pady=15)

start_btn = tk.Button(btn_frame, text="▶ START", font=("Arial", 14, "bold"),
                     bg="#27AE60", fg="white", width=12, height=2,
                     command=start_control, cursor="hand2", relief="raised", bd=3)
start_btn.pack(side="left", padx=10)

stop_btn = tk.Button(btn_frame, text="⏹ STOP", font=("Arial", 14, "bold"),
                    bg="#555555", fg="white", width=12, height=2,
                    command=stop_control, state="disabled", cursor="hand2", relief="raised", bd=3)
stop_btn.pack(side="left", padx=10)

# Footer
footer = tk.Frame(root, bg="#34495E", height=30)
footer.pack(fill="x")

tk.Label(footer, text="Press 'Q' in camera window to close", 
         font=("Arial", 8), bg="#34495E", fg="#BDC3C7").pack(pady=5)

# Load model after UI is ready
root.after(500, load_model_delayed)

root.mainloop()
