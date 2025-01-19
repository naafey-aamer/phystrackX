import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import ImageTk, Image, ImageDraw
from video_processing import VideoProcessor
from utils import resize_frame
import cv2
import numpy as np
from tkinter import ttk
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pytesseract
# from menu1 import MenuScreen
import imageio
from PIL import ImageSequence

class MenuScreen:
    def __init__(self, master):
        self.master = master
        self.master.title("Select Tracking Type")
        self.master.geometry("960x640") 

        self.load_and_display_logo(master)

        # Title label
        tk.Label(master, text="Choose a tracking type:", font=("Helvetica", 16)).pack(pady=(20, 10))

        # Buttons for menu options
        ttk.Button(master, text="Rigid Object Tracking", command=self.on_rigid).pack(fill='x', padx=50, pady=5)
        ttk.Button(master, text="Non-Rigid Object Tracking", command=self.on_non_rigid).pack(fill='x', padx=50, pady=5)

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
        from rigid import VideoApp  # Delayed import
        app = VideoApp(root)
        root.mainloop()

    def on_non_rigid(self):
        self.master.destroy()
        root = tk.Tk()
        root.geometry("960x640")
        app = VideoApp2(root)
        root.mainloop()

class VideoApp2:
    def __init__(self, root):
        self.root = root
        self.root.title("Phys TrackerX")
        
        self.root.geometry("960x740")
        self.processor = VideoProcessor()
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.resize_width = 640 
        self.resize_height = 480 

    def exit_fullscreen(self):
        self.root.attributes('-fullscreen', False)

    def show_progress_bar(self):
        self.progress_popup = tk.Toplevel(self.root)
        self.progress_popup.title("Prcocessing")

        self.progress_label = tk.Label(self.progress_popup, text="Processing frames...")
        self.progress_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.progress_popup, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.progress_popup.update()

    def update_progress_bar(self, value, max_value):
        self.progress_bar["value"] = value
        self.progress_bar["maximum"] = max_value
        self.progress_label.config(text=f"Processing frames... {value}/{max_value}")
        self.progress_popup.update()

    def close_progress_bar(self):
        self.progress_popup.destroy()

    def create_widgets(self):
            # Create a frame to hold the filter widgets on the left side
            self.filter_frame = tk.Frame(self.root)
            self.filter_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
            
            self.open_button = tk.Button(self.filter_frame, text="Open Video", command=self.open_video)
            self.open_button.pack(pady=10)
            
            self.fps_label = tk.Label(self.filter_frame, text="")
            self.fps_label.pack(pady=5)
            
            self.frame_button = tk.Button(self.filter_frame, text="Select Initial and Final Frames", command=self.select_frames)
            self.frame_button.pack(pady=10)

            self.filter_button = tk.Button(self.filter_frame, text="Apply Filters", command=self.show_filter_popup)
            self.filter_button.pack(pady=10)
            
            self.undo_button = tk.Button(self.filter_frame, text="Reset", command=self.undo_filter)
            self.undo_button.pack(pady=10)
            
            self.distance_button = tk.Button(self.filter_frame, text="Set Reference Distance", command=self.set_reference_distance)
            self.distance_button.pack(pady=10)
            
            # self.clip_button = tk.Button(self.filter_frame, text="Clip Video", command=self.clip_video)
            # self.clip_button.pack(pady=10)
            
            self.axis_button = tk.Button(self.filter_frame, text="Mark Axes", command=self.mark_axes)
            self.axis_button.pack(pady=10)
        

            self.track_blob_button = tk.Button(self.filter_frame, text="Track Blob", command=self.track_blob)
            self.track_blob_button.pack(pady=10)

            # self.save_coordinates_button = tk.Button(self.filter_frame, text="Save Coordinates", command=self.save_coordinates)
            # self.save_coordinates_button.pack(pady=10)

            self.mark_pressure_button = tk.Button(self.filter_frame, text="OCR", command=self.mark_pressure_area)
            self.mark_pressure_button.pack(pady=10)
            
            # self.extract_pressure_button = tk.Button(self.filter_frame, text="Extract Pressure Values", command=self.extract_pressure_values)
            # self.extract_pressure_button.pack(pady=10)
            
            self.info_label = tk.Label(self.filter_frame, text="")
            self.info_label.pack(pady=10)
            
            # # Create a button to show tracked points in a table
            # self.table_button = tk.Button(self.filter_frame, text="Show Tracked Points Table", command=self.show_tracked_points_table)
            # self.table_button.pack(pady=10)
            
            self.heatmap_button = tk.Button(self.filter_frame, text="Generate Heatmap", command=self.generate_heatmap)
            self.heatmap_button.pack(pady=10)

            self.trail_button = tk.Button(self.filter_frame, text="Generate Marker Trail", command=self.generate_marker_trail)
            self.trail_button.pack(pady=10)

            self.plot_button = tk.Button(self.filter_frame, text="Plot Centroid Coordinates", command=self.plot_centroid_coordinates)
            self.plot_button.pack(pady=10)

            
            # Create a frame to hold the video and slider widgets on the right side
            self.video_frame = tk.Frame(self.root)
            self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            self.video_view = tk.Canvas(self.video_frame, width=640, height=480)
            self.video_view.pack(pady=20, expand=True)
            
            self.slider = tk.Scale(self.video_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=400, resolution=1, command=self.update_frame)
            self.slider.pack(pady=10)

            self.menu_button = tk.Button(self.filter_frame, text="Back to Menu", command=self.back_to_menu)
            self.menu_button.pack(pady=10)
            

    def open_video(self):
        self.processor.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.MP4")])
        if self.processor.video_path:
            self.processor.fps = int(simpledialog.askinteger("FPS", "Enter FPS:"))
            self.fps_label.config(text=f"FPS: {self.processor.fps}")
            self.load_video()
    
    def load_video(self):
        cap = cv2.VideoCapture(self.processor.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video")
            return

        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read video frame")
            return
        cap.release()

        # Show cropping interface
        self.show_crop_interface(frame)

    def show_crop_interface(self, frame):
        top = tk.Toplevel(self.root)
        top.title("Draw cropping area")
        original_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Compute scaling factor to maintain aspect ratio within 640x640 limit
        max_size = 640
        scale = min(max_size / original_img.width, max_size / original_img.height)
        display_img = original_img.resize((int(original_img.width * scale), int(original_img.height * scale)), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image=display_img)
        canvas = tk.Canvas(top, width=display_img.width, height=display_img.height)
        canvas.pack()
        canvas.create_image(0, 0, image=photo, anchor='nw')
        canvas.image = photo  # Keep a reference!

        self.crop_rectangle = None

        def on_mouse_click(event):
            self.start_x, self.start_y = event.x, event.y
            self.crop_rectangle = canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='red')

        def on_mouse_move(event):
            if self.crop_rectangle:
                canvas.coords(self.crop_rectangle, self.start_x, self.start_y, event.x, event.y)

        def on_mouse_release(event):
            if self.crop_rectangle:
                # Scale coordinates back to original dimensions
                self.crop_coords = (
                    int(self.start_x / scale),
                    int(self.start_y / scale),
                    int(event.x / scale),
                    int(event.y / scale)
                )
                top.destroy()
                self.read_and_crop_video()  # Call to process the video after cropping is set
            else:
                self.crop_coords = None

        def skip_cropping():
            messagebox.showinfo("Skip Cropping", "No cropping will be applied.")
            self.crop_coords = None
            top.destroy()
            self.read_and_crop_video()  # Process without cropping

        canvas.bind("<ButtonPress-1>", on_mouse_click)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_release)

        skip_button = tk.Button(top, text="Skip Cropping", command=skip_cropping)
        skip_button.pack(side=tk.BOTTOM, pady=10)

    def read_and_crop_video(self):
        cap = cv2.VideoCapture(self.processor.video_path)
        self.processor.frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.show_progress_bar()
        self.update_progress_bar(0, total_frames)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.crop_coords:
                x1, y1, x2, y2 = self.crop_coords
                frame = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            frame = resize_frame(frame, 640, 480)
            self.processor.frames.append(frame)

            frame_count += 1
            self.update_progress_bar(frame_count, total_frames)
        cap.release()

        self.close_progress_bar()

        if self.processor.frames:
            self.display_first_frame()

    def display_first_frame(self):
        frame = self.processor.frames[0]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=img)
        self.video_view.create_image(0, 0, image=photo, anchor='nw')
        self.photo = photo
        self.slider['to'] = len(self.processor.frames) - 1

    def update_frame(self, event):
        frame_number = int(self.slider.get())
        if self.processor.filtered_images is not None and len(self.processor.filtered_images) > frame_number:
            frame = self.processor.filtered_images[frame_number]
        else:
            frame = self.processor.frames[frame_number]  # Use frames directly

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=img)

        self.video_view.delete('all')
        self.video_view.create_image(0, 0, image=photo, anchor='nw')
        self.photo = photo

    def select_frames(self):
        self.processor.initial_frame = int(simpledialog.askinteger("Initial Frame", "Enter the initial frame number:"))
        self.processor.final_frame = int(simpledialog.askinteger("Final Frame", "Enter the final frame number:"))
        self.info_label.config(text=f"Initial Frame: {self.processor.initial_frame}, Final Frame: {self.processor.final_frame}")
        self.clip_video()

    def set_reference_distance(self):
        self.processor.line_coords = []
        messagebox.showinfo("Instruction", "Please click two points on the video to set the reference distance.")
        self.video_view.bind("<Button-1>", self.mark_line)
    
    def mark_line(self, event):
        if len(self.processor.line_coords) < 2:
            self.processor.line_coords.append((event.x, event.y))
            if len(self.processor.line_coords) == 2:
                self.video_view.create_line(self.processor.line_coords[0], self.processor.line_coords[1], fill="red", width=2)
                self.processor.ref_distance = simpledialog.askfloat("Reference Distance", "Enter the reference distance in your chosen unit:")
                self.info_label.config(text=f"Reference Distance: {self.processor.ref_distance} units")
    
    def clip_video(self):
        if self.processor.initial_frame is None or self.processor.final_frame is None:
            messagebox.showerror("Error", "Please select initial and final frames first.")
            return

        self.processor.clip_video()

        self.processor.frames = self.processor.cropped_frames.copy()

        if self.processor.filtered_images:
            self.processor.apply_filters_to_clipped_frames()

        self.display_clipped_frames()

    def display_clipped_frames(self):
        def update_display(frame_index):
            if self.processor.filtered_images is not None and len(self.processor.filtered_images) > frame_index:
                frame = self.processor.filtered_images[frame_index]
            else:
                frame = self.processor.frames[frame_index]  # Use frames directly
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=img)
            self.video_view.delete('all')
            self.video_view.create_image(0, 0, image=photo, anchor='nw')
            self.photo = photo

        self.slider['to'] = len(self.processor.frames) - 1  # Use frames directly
        self.slider.config(command=lambda event: update_display(int(self.slider.get())))
        update_display(0)

    def mark_axes(self):
        self.processor.axis_coords = []
        messagebox.showinfo("Instruction", "Please click to set the origin of the axes.")
        self.video_view.bind("<Button-1>", self.set_origin)

    def set_origin(self, event):
        self.processor.axis_coords = [(event.x, event.y)]  # Origin point
        self.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="blue", width=2)
        self.video_view.unbind("<Button-1>")
        messagebox.showinfo("Instruction", "Click Again To Finalize The Axes")
        self.video_view.bind("<Motion>", self.update_axes_image)
        self.video_view.bind("<Button-1>", self.drop_axes)

    def update_axes_image(self, event):
        if len(self.processor.axis_coords) == 1:
            origin = self.processor.axis_coords[0]
            if self.processor.axis_image_id:
                self.video_view.delete(self.processor.axis_image_id)
            
            img = Image.new('RGBA', (640, 480), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw axes extending to the borders of the image/frame
            # X-axis
            draw.line([(0, origin[1]), (640, origin[1])], fill="blue", width=2)
            # Y-axis
            draw.line([(origin[0], 0), (origin[0], 480)], fill="green", width=2)
            
            self.processor.axis_image = ImageTk.PhotoImage(img)
            self.processor.axis_image_id = self.video_view.create_image(0, 0, image=self.processor.axis_image, anchor='nw')

    def drop_axes(self, event):
        if len(self.processor.axis_coords) == 1:
            origin = self.processor.axis_coords[0]
            self.processor.axis_coords.append((640, origin[1]))  # Positive X-axis end point
            self.processor.axis_coords.append((0, origin[1]))    # Negative X-axis end point
            self.processor.axis_coords.append((origin[0], 480))  # Positive Y-axis end point
            self.processor.axis_coords.append((origin[0], 0))    # Negative Y-axis end point

            # Draw axes to the borders of the image/frame
            self.video_view.create_line(0, origin[1], 640, origin[1], fill="blue", width=2)  # X-axis
            self.video_view.create_line(origin[0], 0, origin[0], 480, fill="green", width=2)  # Y-axis
            
            self.video_view.unbind("<Motion>")
            self.video_view.unbind("<Button-1>")


    def mark_points_to_track(self):
        self.processor.points_to_track = []
        messagebox.showinfo("Instruction", "Please click points on the video to mark them for tracking.")
        self.video_view.bind("<Button-1>", self.mark_point)
    
    def mark_point(self, event):
        self.processor.points_to_track.append((event.x, event.y))
        self.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
    
    def start_tracking(self):
        """
        Implements Lucas-Kanade optical flow tracking optimized for non-rigid body tracking.
        
        This implementation differs from rigid tracking in that it:
        - Allows for deformation between tracked points
        - Doesn't assume constant distance between points
        - Tracks each point independently without rigid body constraints
        - Suitable for tracking deformable objects, fluids, or multiple independent points
        """

        # Input validation: ensure video has been preprocessed
        if not self.processor.frames:
            messagebox.showerror("Error", "Please clip the video first.")
            return
        
        # Input validation: ensure tracking points have been marked
        # For non-rigid tracking, points can be scattered across deformable regions
        if not self.processor.points_to_track:
            messagebox.showerror("Error", "Please mark points to track first.")
            return

        # Convert tracking points to numpy array
        # Shape: Nx1x2 where:
        # N = number of points
        # 1 = single coordinate
        # 2 = x,y coordinates
        p0 = np.array(self.processor.points_to_track, dtype=np.float32).reshape(-1, 1, 2)

        # Configure Lucas-Kanade parameters for non-rigid tracking
        lk_params = dict(
            winSize=(15, 15),      # Search window size per pyramid level
                                # Smaller than rigid tracking to capture local deformations
            
            maxLevel=2,            # Number of pyramid levels
                                # Balances between large movements and local accuracy
            
            criteria=(             # Termination criteria
                cv2.TERM_CRITERIA_EPS |      # Stop on accuracy threshold
                cv2.TERM_CRITERIA_COUNT,     # Stop on max iterations
                10,                          # Maximum iterations
                0.03                         # Minimum accuracy (epsilon)
            )
        )

        # Process first frame - convert to grayscale if needed
        # Non-rigid tracking requires consistent grayscale processing
        if self.processor.frames[0].ndim == 3 and self.processor.frames[0].shape[2] == 3:
            old_gray = cv2.cvtColor(self.processor.frames[0], cv2.COLOR_BGR2GRAY)
        else:
            old_gray = self.processor.frames[0]  # Already in grayscale

        # Initialize tracking history
        # Dictionary structure:
        # - Keys: point indices
        # - Values: list of (x,y) coordinates representing point trajectory
        points_tracked = {i: [p] for i, p in enumerate(self.processor.points_to_track)}

        # Main tracking loop - process each frame after the first
        for frame in self.processor.frames[1:]:
            # Convert current frame to grayscale if needed
            # Consistent grayscale processing is crucial for accurate tracking
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame

            # Calculate optical flow using Lucas-Kanade method
            # Returns:
            # p1 - New point positions
            # st - Status array (1 = tracked, 0 = lost)
            # err - Error measure for each point
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Filter points based on status
            # Only keep points that were successfully tracked
            good_new = p1[st == 1]  # New positions of good points
            good_old = p0[st == 1]  # Previous positions for reference

            # Update tracking history for each point
            # Each point moves independently in non-rigid tracking
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()  # Extract x,y coordinates
                points_tracked[i].append((a, b))  # Add to trajectory

            # Prepare for next iteration
            old_gray = frame_gray.copy()  # Update previous frame
            p0 = good_new.reshape(-1, 1, 2)  # Update previous points

        # Visualize tracking results
        # Draw points on each frame to show trajectories
        for i, frame in enumerate(self.processor.frames):
            for j, point in points_tracked.items():
                if i < len(point):  # Check if point exists in this frame
                    x, y = point[i]
                    # Draw green circle at point position
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Store tracked points for analysis
        # Non-rigid analysis may include deformation metrics
        self.processor.points_tracked = points_tracked

        # Update UI and display
        self.display_clipped_frames()
        self.plot_button.config(state=tk.NORMAL)  # Enable plotting functionality


    def plot_distances(self):
        if not hasattr(self.processor, 'points_tracked'):
            messagebox.showerror("Error", "No tracked points available. Please start tracking first.")
            return
        
        self.processor.plot_distances()
    

    def show_filter_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Select Filters")
        
        filters = [
            "GMM With Background",
            "GMM Without Background",
            "Optical Flow",
            "Low-pass Filter",
            "High-pass Filter",
            "Median Filter",
            "Exponential Smoothing",
            "Edge Detection Filter",
            "Bilateral Filter",
            "Object Separation",
            "Contrast Adjustment"
        ]
        
        self.filter_vars = {filter_name: tk.BooleanVar() for filter_name in filters}
        
        for filter_name in filters:
            frame = tk.Frame(popup)
            frame.pack(anchor='w', fill='x')
            
            chk = tk.Checkbutton(frame, text=filter_name, variable=self.filter_vars[filter_name])
            chk.pack(side=tk.LEFT, anchor='w')
            
            info_button = tk.Button(frame, text="i", command=lambda fn=filter_name: self.show_filter_preview(fn))
            info_button.pack(side=tk.RIGHT, padx=5)

        apply_button = tk.Button(popup, text="Apply", command=lambda: self.apply_filters(popup))
        apply_button.pack(pady=10)
    
    def show_filter_preview(self, filter_name):
        if not self.processor.frames:
            messagebox.showerror("Error", "Please load a video first.")
            return

        preview_popup = tk.Toplevel(self.root)
        preview_popup.title(f"Preview of {filter_name}")

        middle_frame_index = len(self.processor.frames) // 2
        start_index = max(0, middle_frame_index - 10)
        end_index = min(len(self.processor.frames), middle_frame_index + 10)
        
        frames = []
        prev_frame = None
        for i in range(start_index, end_index):
            frame = self.processor.frames[i].copy()
            frame = self.processor.apply_filter_to_frame(frame, filter_name, prev_frame)
            if filter_name in ["Exponential Smoothing", "Optical Flow", "GMM With Background", "GMM Without Background"]:
                prev_frame = frame
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        gif_path = "/tmp/preview.gif"
        imageio.mimsave(gif_path, frames, fps=5)
        
        gif = Image.open(gif_path)
        frames = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(gif)]
        
        canvas = tk.Canvas(preview_popup, width=gif.width, height=gif.height)
        canvas.pack()
        
        def animate_gif(frame_idx=0):
            if frame_idx < len(frames):
                canvas.create_image(0, 0, image=frames[frame_idx], anchor='nw')
                preview_popup.after(100, animate_gif, (frame_idx + 1) % len(frames))
        
        animate_gif()


    def apply_filters(self, popup):
        selected_filters = [filter_name for filter_name, var in self.filter_vars.items() if var.get()]
        if not selected_filters:
            messagebox.showerror("Error", "No filters selected")
            return

        popup.destroy()
        self.show_progress_bar()

        total_frames = len(self.processor.frames)
        prev_frame = None
        self.processor.filtered_images = []
        for i, frame in enumerate(self.processor.frames):
            for filter_type in selected_filters:
                frame = self.processor.apply_filter_to_frame(frame, filter_type, prev_frame)
            self.processor.filtered_images.append(frame)
            prev_frame = frame
            self.update_progress_bar(i + 1, total_frames)

        self.close_progress_bar()
        self.processor.frames = self.processor.filtered_images.copy()  # Ensure this line exists
        self.update_frame(None)  # Refresh display to show filtered frames


    def undo_filter(self):
        # Clear filtered images
        self.processor.filtered_images = None

        self.load_video()  # Ensure this method reloads the video frames correctly

        self.slider.set(0)
        self.update_frame(None)

    def translate_to_real_coordinates(self, points):
        origin = self.processor.axis_coords[0]
        ref_pixel_dist = np.linalg.norm(np.array(self.processor.line_coords[0]) - np.array(self.processor.line_coords[1]))
        scale_factor = self.processor.ref_distance / ref_pixel_dist
        
        real_coords = []
        for point in points:
            x_real = (point[0] - origin[0]) * scale_factor
            y_real = (origin[1] - point[1]) * scale_factor
            real_coords.append((x_real, y_real))
        
        return real_coords


    def show_tracked_points_table(self):
        if not hasattr(self.processor, 'points_tracked'):
            messagebox.showerror("Error", "No tracked points available. Please start tracking first.")
            return

        table_popup = tk.Toplevel(self.root)
        table_popup.title("Tracked Points Table")
        table_frame = tk.Frame(table_popup)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ["Frame"] + [f"Point {i+1} (x, y)" for i in range(len(self.processor.points_to_track))]
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')

        for col in columns:
            tree.heading(col, text=col)

        max_len = max(len(points) for points in self.processor.points_tracked.values())
        for i in range(max_len):
            row = [i]
            for j in range(len(self.processor.points_to_track)):
                if i < len(self.processor.points_tracked[j]):
                    x, y = self.processor.points_tracked[j][i]
                    row.append(f"({int(x)}, {int(y)})")
                else:
                    row.append("")
            tree.insert('', tk.END, values=row)

        tree.pack(fill=tk.BOTH, expand=True)

        # Add the export button
        export_button = tk.Button(table_popup, text="Export as CSV", command=self.export_tracked_points_to_csv)
        export_button.pack(pady=10)


    def export_tracked_points_to_csv(self):
        if not hasattr(self.processor, 'points_tracked'):
            messagebox.showerror("Error", "No tracked points available to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            return

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            columns = ["Frame"] + [f"Point {i+1} (x, y)" for i in range(len(self.processor.points_to_track))]
            writer.writerow(columns)

            max_len = max(len(points) for points in self.processor.points_tracked.values())
            for i in range(max_len):
                row = [i]
                for j in range(len(self.processor.points_to_track)):
                    if i < len(self.processor.points_tracked[j]):
                        real_coords = self.translate_to_real_coordinates(self.processor.points_tracked[j])
                        x, y = real_coords[i]
                        row.append(f"({x:.2f}, {y:.2f})")
                    else:
                        row.append("")
                writer.writerow(row)

    def track_blob(self):
        messagebox.showinfo("Instruction", "Please draw a bounding box around the area to track.")
        self.video_view.bind("<Button-1>", self.start_shape)

    def start_shape(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.video_view.bind("<B1-Motion>", self.draw_shape)
        self.video_view.bind("<ButtonRelease-1>", self.finish_shape)

    def draw_shape(self, event):
        self.video_view.delete("current_shape")
        self.video_view.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", tag="current_shape")

    def finish_shape(self, event):
        self.video_view.unbind("<B1-Motion>")
        self.video_view.unbind("<ButtonRelease-1>")
        self.video_view.delete("current_shape")
        self.end_x, self.end_y = event.x, event.y
        print(f"Bounding box: ({self.start_x}, {self.start_y}) to ({self.end_x}, {self.end_y})")  # Debug statement
        self.perform_blob_tracking()

    def save_area_time_graph(self):
        if not hasattr(self.processor, 'fps') or self.processor.fps is None:
            messagebox.showerror("Error", "FPS value is missing.")
            return

        # Ensure contour_points are not empty and are in the correct format
        areas = []
        for contour in self.processor.contour_points:
            if len(contour) > 0:  # Ensure the contour is not empty
                contour_array = np.array(contour, dtype=np.float32)  # Convert to correct format
                areas.append(cv2.contourArea(contour_array))
            else:
                areas.append(0)  # Handle case where contour is empty

        # Calculate time for each frame based on FPS
        times = [frame_num / self.processor.fps for frame_num in range(len(areas))]

        # Plot the area/time graph
        plt.figure()
        plt.plot(times, areas, label="Contour Area")
        plt.xlabel('Time (s)')
        plt.ylabel('Area')
        plt.title('Tracked Contour Area Over Time')
        plt.legend()
        plt.show()

        # Ask where to save the graph
        # file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        # if file_path:
        #     plt.savefig(file_path)
        #     messagebox.showinfo("Success", f"Graph saved at {file_path}")

        # Ask where to save the CSV file
        csv_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if csv_file_path:
            # Save the area and time data to a CSV file
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time (s)", "Area"])
                for time, area in zip(times, areas):
                    writer.writerow([time, area])
            
            messagebox.showinfo("Success", f"Data saved to {csv_file_path}")



    def perform_blob_tracking(self):
        print("Starting blob tracking...")  # Debug statement
        self.processor.tracked_points = []
        self.processor.contour_points = []
        resize_width, resize_height = 640, 480

        # Ensure bbox coordinates are within frame dimensions
        start_x = min(self.start_x, self.end_x)
        start_y = min(self.start_y, self.end_y)
        end_x = max(self.start_x, self.end_x)
        end_y = max(self.start_y, self.end_y)

        bbox = (
            max(0, int(start_x * resize_width / self.processor.frames[0].shape[1])),
            max(0, int(start_y * resize_height / self.processor.frames[0].shape[0])),
            max(0, int(end_x * resize_width / self.processor.frames[0].shape[1]) - int(start_x * resize_width / self.processor.frames[0].shape[1])),
            max(0, int(end_y * resize_height / self.processor.frames[0].shape[0]) - int(start_y * resize_height / self.processor.frames[0].shape[0]))
        )

        print(f"Adjusted bounding box: {bbox}")  # Debug statement

        for frame_index, frame in enumerate(self.processor.frames):
            frame = cv2.resize(frame, (resize_width, resize_height))
            x, y, w, h = bbox
            roi = frame[y:y+h, x:x+w]

            if roi.size == 0:
                print(f"Empty ROI at frame {frame_index}")  # Debug statement
                continue

            if len(roi.shape) == 2 or roi.shape[2] == 1:
                gray = roi  # Already grayscale
            else:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(frame[y:y+h, x:x+w], [max_contour], -1, (0, 255, 0), 2)
                cX, cY = self.calculate_centroid(max_contour)
                if cX is not None and cY is not None:
                    try:
                        self.processor.tracked_points.append((cX + x, cY + y))
                        self.processor.contour_points.append([(point[0][0] + x, point[0][1] + y) for point in max_contour])
                        cv2.circle(frame[y:y+h, x:x+w], (cX, cY), 5, (255, 0, 0), -1)
                        print(f"Frame {frame_index}: Centroid at ({cX + x}, {cY + y})")  # Debug statement
                    except Exception as e:
                        print(f"Exception while appending centroid at frame {frame_index}: {e}")
                else:
                    print(f"Centroid calculation returned None at frame {frame_index}")  # Debug statement
            else:
                print(f"No contours found at frame {frame_index}")  # Debug statement

            # Update the frame in the processor's frame list
            self.processor.frames[frame_index] = frame

        self.display_clipped_frames()
        print("Blob tracking completed.") 

        # Convert tracked points to real coordinates
        real_tracked_points = self.translate_to_real_coordinates(self.processor.tracked_points)
        real_contour_points = [self.translate_to_real_coordinates(contour) for contour in self.processor.contour_points]
        self.processor.tracked_points = real_tracked_points
        self.processor.contour_points = real_contour_points

        if messagebox.askyesno("Save Coordinates", "Blob tracking completed. Would you like to save the centroid and blob coordinates?"):
            self.save_coordinates()

        
        # Once the blob tracking is complete, ask if the user wants to save the area/time graph
        if messagebox.askyesno("Save Area/Time Graph", "Would you like to plot the area-time graph and save the coordinates of the blob?"):
            self.save_area_time_graph()


    def calculate_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
        return None, None

    def generate_heatmap(self):
        if not hasattr(self.processor, 'tracked_points') or not self.processor.tracked_points:
            messagebox.showerror("Error", "No coordinates to generate heatmap.")
            return
        resize_width, resize_height = 640, 480
        heatmap = np.zeros((resize_height, resize_width), dtype=np.float32)
        
        # Ensure bbox coordinates are within frame dimensions
        start_x = min(self.start_x, self.end_x)
        start_y = min(self.start_y, self.end_y)
        end_x = max(self.start_x, self.end_x)
        end_y = max(self.start_y, self.end_y)

        bbox = (
            max(0, int(start_x * resize_width / self.processor.frames[0].shape[1])),
            max(0, int(start_y * resize_height / self.processor.frames[0].shape[0])),
            max(0, int(end_x * resize_width / self.processor.frames[0].shape[1]) - int(start_x * resize_width / self.processor.frames[0].shape[1])),
            max(0, int(end_y * resize_height / self.processor.frames[0].shape[0]) - int(start_y * resize_height / self.processor.frames[0].shape[0]))
        )
        
        print(f"Adjusted bounding box for heatmap: {bbox}")

        for i, (x, y) in enumerate(self.processor.tracked_points):
            frame = self.processor.frames[i].copy()
            roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                cX, cY = self.calculate_centroid(max_contour)
                cv2.circle(roi, (cX, cY), 5, (255, 0, 0), -1)
                heatmap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = cv2.add(
                    heatmap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]],
                    thresh.astype(np.float32)
                )

        heatmap_cropped = heatmap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        heatmap_normalized = cv2.normalize(heatmap_cropped, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        plt.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        plt.title('Heatmap of Tracked Blob Coordinates')
        plt.show()

    def generate_marker_trail(self):
        if not hasattr(self.processor, 'tracked_points') or not self.processor.tracked_points:
            messagebox.showerror("Error", "No coordinates to generate marker trail.")
            return

        interval = simpledialog.askinteger("Marker Interval", "Enter the interval between markers (in frames):", minvalue=1)
        if interval is None:
            return

        snapshots = []
        for i, (x, y) in enumerate(self.processor.tracked_points):
            if i % interval == 0:
                frame = self.processor.frames[i].copy()
                # Ensure (x, y) are integers
                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                snapshots.append(frame)

        # Determine the size of the collage
        num_snapshots = len(snapshots)
        collage_width = min(4, num_snapshots)  # Number of columns
        collage_height = (num_snapshots + 3) // 4  # Number of rows

        # Get dimensions of a single frame
        frame_height, frame_width, _ = snapshots[0].shape

        # Create an empty collage image
        collage = np.zeros((collage_height * frame_height, collage_width * frame_width, 3), dtype=np.uint8)

        # Place snapshots into the collage
        for idx, snapshot in enumerate(snapshots):
            row = idx // collage_width
            col = idx % collage_width
            start_y = row * frame_height
            start_x = col * frame_width
            collage[start_y:start_y + frame_height, start_x:start_x + frame_width] = snapshot

        # Display the collage
        plt.imshow(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))
        plt.title('Marker Trail of Tracked Blob')
        plt.show()



    def plot_centroid_coordinates(self):
        if not hasattr(self.processor, 'tracked_points') or not self.processor.tracked_points:
            messagebox.showerror("Error", "No coordinates to plot.")
            return

        # x_coords = [(p[0] - self.processor.axis_coords[0][0]) * self.processor.ref_distance for p in self.processor.tracked_points]
        # y_coords = [(p[1] - self.processor.axis_coords[0][1]) * self.processor.ref_distance for p in self.processor.tracked_points]

        # real_tracked_points = self.translate_to_real_coordinates(self.processor.tracked_points)
        # self.processor.tracked_points = real_tracked_points

        x_coords = [p[0] for p in self.processor.tracked_points]
        y_coords = [p[1] for p in self.processor.tracked_points]

        plt.figure()
        plt.plot(range(len(x_coords)), x_coords, label='X Coordinates')
        plt.plot(range(len(y_coords)), y_coords, label='Y Coordinates')
        plt.xlabel('Frame')
        plt.ylabel('Coordinate (Real Units)')
        plt.title('Centroid Coordinates Over Time')
        plt.legend()
        plt.show()


    def save_coordinates(self):
        if not hasattr(self.processor, 'tracked_points') or not self.processor.tracked_points:
            messagebox.showerror("Error", "No coordinates to save.")
            return

        if not hasattr(self.processor, 'ref_distance') or self.processor.ref_distance is None:
            messagebox.showerror("Error", "Reference distance not set.")
            return

        if not hasattr(self.processor, 'axis_coords') or len(self.processor.axis_coords) < 1:
            messagebox.showerror("Error", "Axis origin not set.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            return

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Centroid X (real units)", "Centroid Y (real units)", "Contour Points (real units)"])
            for i, (centroid, contour) in enumerate(zip(self.processor.tracked_points, self.processor.contour_points)):
                contour_str = "; ".join([f"({x:.2f}, {y:.2f})" for x, y in contour])
                writer.writerow([i, centroid[0], centroid[1], contour_str])



    def mark_pressure_area(self):
        messagebox.showinfo("Instruction", "Please draw a bounding box around the values.")
        self.video_view.bind("<Button-1>", self.start_pressure_box)

    def start_pressure_box(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.video_view.bind("<B1-Motion>", self.draw_pressure_box)
        self.video_view.bind("<ButtonRelease-1>", self.finish_pressure_box)

    def draw_pressure_box(self, event):
        self.video_view.delete("pressure_box")
        self.video_view.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red", tag="pressure_box")

    def finish_pressure_box(self, event):
        self.video_view.unbind("<B1-Motion>")
        self.video_view.unbind("<ButtonRelease-1>")
        self.pressure_box = (self.start_x, self.start_y, event.x, event.y)
        print(f" bounding box: {self.pressure_box}")
        self.extract_pressure_values()

    def preprocess_image_for_ocr(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize the image to double its original size
        gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        # Apply dilation and erosion to enhance characters
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
        # Apply denoising
        processed_image = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)
        return processed_image

    def extract_pressure_values(self):
        if not hasattr(self, 'pressure_box'):
            messagebox.showerror("Error", "Pressure bounding box not set.")
            return

        start_x, start_y, end_x, end_y = self.pressure_box
        pressure_values = []

        total_frames = len(self.processor.frames)
        self.show_progress_bar()
        self.update_progress_bar(0, total_frames)

        for frame_index, frame in enumerate(self.processor.frames):
            roi = frame[start_y:end_y, start_x:end_x]
            processed_roi = self.preprocess_image_for_ocr(roi)
            config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789m."
            text = pytesseract.image_to_string(processed_roi, config=config)
            pressure_values.append(text.strip())
            self.update_progress_bar(frame_index + 1, total_frames)

        self.close_progress_bar()

        # Optionally save the pressure values to a file
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Frame", "Pressure Value"])
                for i, value in enumerate(pressure_values):
                    writer.writerow([i, value])

        # Display the first few values for confirmation
        for i, value in enumerate(pressure_values[:10]):
            print(f"Frame {i}: Pressure value: {value}")

    def back_to_menu(self):
        self.root.destroy()
        root = tk.Tk()
        root.geometry("960x640")
        menu_x = MenuScreen(root)
        root.mainloop()

    def on_close(self):
        self.root.destroy()
