import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import filedialog, simpledialog, messagebox
from video_processing import display_clipped_frames
import tkinter as tk
def start_tracking(app):
    if not app.cropped_frames:
        messagebox.showerror("Error", "Please clip the video first.")
        return
    
    if not app.points_to_track:
        messagebox.showerror("Error", "Please mark points to track first.")
        return
    
    p0 = np.array(app.points_to_track, dtype=np.float32).reshape(-1, 1, 2)
    
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    old_gray = cv2.cvtColor(app.cropped_frames[0], cv2.COLOR_BGR2GRAY)
    points_tracked = {i: [p] for i, p in enumerate(app.points_to_track)}
    
    for frame in app.cropped_frames[1:]:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            points_tracked[i].append((a, b))
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    for i, frame in enumerate(app.cropped_frames):
        for j, point in points_tracked.items():
            if i < len(point):
                x, y = point[i]
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    display_clipped_frames(app)
    
    for i, points in points_tracked.items():
        print(f"Point {i}:")
        for point in points:
            print(f"    {point}")
    
    app.points_tracked = points_tracked
    app.plot_button.config(state=tk.NORMAL)

def plot_distances(app):
    if not hasattr(app, 'points_tracked'):
        messagebox.showerror("Error", "No tracked points available. Please start tracking first.")
        return
    
    origin = app.axis_coords[0]
    x_axis_end = app.axis_coords[1]
    y_axis_end = app.axis_coords[2]
    
    ref_pixel_dist = np.linalg.norm(np.array(app.line_coords[0]) - np.array(app.line_coords[1]))
    scale_factor = app.ref_distance / ref_pixel_dist
    
    for i, points in app.points_tracked.items():
        x_coords = [(p[0] - origin[0]) * scale_factor for p in points]
        y_coords = [(origin[1] - p[1]) * scale_factor for p in points]
        
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x_coords, label=f'Point {i} X')
        plt.xlabel('Frame')
        plt.ylabel('X Distance')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(y_coords, label=f'Point {i} Y')
        plt.xlabel('Frame')
        plt.ylabel('Y Distance')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
