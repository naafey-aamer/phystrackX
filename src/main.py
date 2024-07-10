import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import ImageTk, Image, ImageDraw
from video_processing import VideoProcessor
from utils import resize_frame
import cv2
import numpy as np
from tkinter import ttk
import csv
from nonrigid import VideoApp2
from rigid import VideoApp
from auto import VideoApp3

class WelcomeScreen:
    def __init__(self, master):
        self.master = master
        self.master.title("Welcome")
        self.master.geometry("960x640")  

        # Title label
        self.label = tk.Label(master, text="Welcome to PhysTrackX", font=("Helvetica", 24))
        self.label.pack(pady=(20, 0))

        # Subtitle label
        subtitle_label = tk.Label(master, text="A Project By", font=("Helvetica", 18))
        subtitle_label.pack(pady=(10, 0))

        # Load and display the image
        self.load_and_display_image(master)

        self.animate_text()

    def load_and_display_image(self, master):
        # Load the image
        image_path = "physlab_logo.png"
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)

        # Create a label to display the image
        image_label = tk.Label(master, image=photo)
        image_label.image = photo  # Keep a reference, prevent GC
        image_label.pack(pady=(10, 20))

    def animate_text(self):
        text = self.label.cget("text")
        if text.endswith("..."):
            self.label.config(text="Welcome to PhysTrackX")
        else:
            self.label.config(text=text + ".")
        self.master.after(500, self.animate_text)

class MenuScreen:
    def __init__(self, master):
        self.master = master
        self.master.title("Select Tracking Type")
        self.master.geometry("960x640") 

        # Load and display the logo, resized to fit the window
        self.load_and_display_logo(master)

        # Title label
        tk.Label(master, text="Choose a tracking type:", font=("Helvetica", 16)).pack(pady=(20, 10))

        # Buttons for menu options
        ttk.Button(master, text="Rigid Object Tracking", command=self.on_rigid).pack(fill='x', padx=50, pady=5)
        ttk.Button(master, text="Non-Rigid Object Tracking", command=self.on_non_rigid).pack(fill='x', padx=50, pady=5)
        # ttk.Button(master, text="Auto Tracking", command=self.on_auto).pack(fill='x', padx=50, pady=5)

    def load_and_display_logo(self, master):
        # Load the image
        image_path = "phys_track_logo.png" 
        image = Image.open(image_path)

        # Resize the image to fit the window width while maintaining aspect ratio
        base_width = 700  # Set width smaller than the window width for padding considerations
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image = image.resize((base_width, h_size), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image)

        # Create a label to display the image
        image_label = tk.Label(master, image=photo)
        image_label.image = photo  # Keep a reference, prevent GC
        image_label.pack(pady=(10, 0))

    def on_rigid(self):
        self.master.destroy()
        root = tk.Tk()
        root.geometry("960x640")
        app = VideoApp(root)
        root.mainloop()

    def on_non_rigid(self):
        self.master.destroy()
        root = tk.Tk()
        root.geometry("960x640")
        app = VideoApp2(root)
        root.mainloop()

    def on_auto(self):
        self.master.destroy()
        root = tk.Tk()
        root.geometry("960x640")
        app = VideoApp3(root)
        root.mainloop()


def main():
    root = tk.Tk()
    root.geometry("960x640")
    welcome = WelcomeScreen(root)
    root.after(2500, root.destroy)  # Close welcome screen after 5 seconds
    root.mainloop()
    
    root = tk.Tk()
    root.geometry("960x640")
    menu = MenuScreen(root)
    root.mainloop()

if __name__ == "__main__":
    main()
