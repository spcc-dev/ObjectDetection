import tkinter as tk
import pyttsx3
import threading

def speak_text(text):
    """Create a fresh engine each time to allow multiple speeches."""
    engine = pyttsx3.init()  # new engine for each call
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def on_button_click():
    text = "Hello! You clicked the button."
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()

# Tkinter GUI
root = tk.Tk()
root.title("Async TTS Example")
root.geometry("300x150")

button = tk.Button(root, text="Speak", command=on_button_click)
button.pack(pady=50)

root.mainloop()
