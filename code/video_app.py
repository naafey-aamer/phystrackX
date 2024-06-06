import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from widgets import create_widgets
from video_processing import load_video, clip_video, update_frame, set_reference_distance, mark_axes, mark_points_to_track
from tracking import start_tracking, plot_distances

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phys Tracker")
        
        # Initialize variables
        self.video_path = ""
        self.fps = None
        self.initial_frame = None
        self.final_frame = None
        self.ref_distance = None
        self.line_coords = []
        self.axis_coords = []
        self.cropped_frames = []
        self.axis_image = None
        self.axis_image_id = None
        self.points_to_track = []
        
        # Create widgets
        create_widgets(self)
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def on_close(self):
        self.root.destroy()
