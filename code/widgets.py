import tkinter as tk
from video_processing import open_video, update_frame, select_frames, set_reference_distance, mark_axes, mark_points_to_track, clip_video
from tracking import start_tracking, plot_distances

def create_widgets(app):
    app.open_button = tk.Button(app.root, text="Open Video", command=lambda: open_video(app))
    app.open_button.pack(pady=10)
    
    app.fps_label = tk.Label(app.root, text="")
    app.fps_label.pack(pady=5)
    
    app.video_view = tk.Canvas(app.root, width=640, height=480)
    app.video_view.pack(pady=20)
    
    app.slider = tk.Scale(app.root, from_=0, to=100, orient=tk.HORIZONTAL, length=400, resolution=1, command=lambda event: update_frame(app, event))
    app.slider.pack(pady=10)
    
    app.frame_button = tk.Button(app.root, text="Select Initial and Final Frames", command=lambda: select_frames(app))
    app.frame_button.pack(pady=10)
    
    app.distance_button = tk.Button(app.root, text="Set Reference Distance", command=lambda: set_reference_distance(app))
    app.distance_button.pack(pady=10)
    
    app.clip_button = tk.Button(app.root, text="Clip Video", command=lambda: clip_video(app))
    app.clip_button.pack(pady=10)
    
    app.axis_button = tk.Button(app.root, text="Mark Axes", command=lambda: mark_axes(app))
    app.axis_button.pack(pady=10)
    
    app.track_button = tk.Button(app.root, text="Mark Points to Track", command=lambda: mark_points_to_track(app))
    app.track_button.pack(pady=10)
    
    app.track_start_button = tk.Button(app.root, text="Start Tracking", command=lambda: start_tracking(app))
    app.track_start_button.pack(pady=10)
    
    app.plot_button = tk.Button(app.root, text="Plot X and Y Distances", command=lambda: plot_distances(app))
    app.plot_button.pack(pady=10)
    app.plot_button.config(state=tk.DISABLED)
    
    app.info_label = tk.Label(app.root, text="")
    app.info_label.pack(pady=10)
