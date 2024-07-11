import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import ImageTk, Image, ImageDraw
from video_processing import VideoProcessor
from utils import resize_frame
import cv2
import numpy as np
from tkinter import ttk
import csv
from matplotlib import pyplot as plt
import imageio
from PIL import ImageSequence

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
        from nonrigid import VideoApp2  # Delayed import
        root.geometry("960x640")
        app = VideoApp2(root)
        root.mainloop()

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phys TrackerX")
        
        # Make the window cover the entire screen
        self.root.geometry("960x640")
        self.processor = VideoProcessor()
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def exit_fullscreen(self):
        self.root.attributes('-fullscreen', False)

    def show_progress_bar(self):
        self.progress_popup = tk.Toplevel(self.root)
        self.progress_popup.title("Applying Filters")

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
            
            self.filter_button = tk.Button(self.filter_frame, text="Apply Filters", command=self.show_filter_popup)
            self.filter_button.pack(pady=10)
            
            self.undo_button = tk.Button(self.filter_frame, text="Reset", command=self.undo_filter)
            self.undo_button.pack(pady=10)
            
            self.frame_button = tk.Button(self.filter_frame, text="Select Initial and Final Frames", command=self.select_frames)
            self.frame_button.pack(pady=10)
            
            self.distance_button = tk.Button(self.filter_frame, text="Set Reference Distance", command=self.set_reference_distance)
            self.distance_button.pack(pady=10)
            
            # self.clip_button = tk.Button(self.filter_frame, text="Clip Video", command=self.clip_video)
            # self.clip_button.pack(pady=10)
            
            self.axis_button = tk.Button(self.filter_frame, text="Mark Axes", command=self.check_reference_distance)
            self.axis_button.pack(pady=10)
            
            self.track_button = tk.Button(self.filter_frame, text="Mark Points to Track", command=self.choose_tracking_method)
            self.track_button.pack(pady=10)

            self.track_start_button = tk.Button(self.filter_frame, text="Start Tracking", command=self.start_tracking)
            self.track_start_button.pack(pady=10)
            
            self.track_coords_button = tk.Button(self.filter_frame, text="Tracked Coordinates", command=self.show_tracked_coordinates_window)
            self.track_coords_button.pack(pady=10)
            self.track_coords_button.config(state=tk.DISABLED)  # Disable the button initially

            self.info_label = tk.Label(self.filter_frame, text="")
            self.info_label.pack(pady=10)
            
            # Create a button to show tracked points in a table
            # self.table_button = tk.Button(self.filter_frame, text="Show Tracked Points Table", command=self.show_tracked_points_table)
            # self.table_button.pack(pady=10)
            
            # Create a frame to hold the video and slider widgets on the right side
            self.video_frame = tk.Frame(self.root)
            self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            self.video_view = tk.Canvas(self.video_frame, width=640, height=480)
            self.video_view.pack(pady=20, expand=True)
            
            self.slider = tk.Scale(self.video_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=400, resolution=1, command=self.update_frame)
            self.slider.pack(pady=10)

            self.menu_button = tk.Button(self.filter_frame, text="Back to Menu", command=self.back_to_menu)
            self.menu_button.pack(pady=10)

            # self.filter_frame2 = tk.Frame(self.root)

            # self.filter_frame2.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

            # self.open_button = tk.Button(self.filter_frame2, text="Ass", command=self.open_video)
            # self.open_button.pack(pady=10)
    


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

        # Replace original frames with the clipped frames
        self.processor.frames = self.processor.cropped_frames.copy()

        # If there are filtered images, apply the filters to the clipped frames
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

    def check_reference_distance(self):
        if not hasattr(self.processor, 'ref_distance') or self.processor.ref_distance is None:
            messagebox.showerror("Error", "Please set the reference distance before marking axes.")
        else:
            self.mark_axes()

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

    def mark_points_to_track(self):
        self.processor.points_to_track = []
        messagebox.showinfo("Instruction", "Please click points on the video to mark them for tracking.")
        self.video_view.bind("<Button-1>", self.mark_point)
    
    def mark_point(self, event):
        self.processor.points_to_track.append((event.x, event.y))
        self.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
    
    def start_tracking(self):
        if not self.processor.frames:
            messagebox.showerror("Error", "Please clip the video first.")
            return
        
        if not self.processor.points_to_track:
            messagebox.showerror("Error", "Please mark points to track first.")
            return
        
        p0 = np.array(self.processor.points_to_track, dtype=np.float32).reshape(-1, 1, 2)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Check if the first frame is already grayscale
        if self.processor.frames[0].ndim == 3 and self.processor.frames[0].shape[2] == 3:
            old_gray = cv2.cvtColor(self.processor.frames[0], cv2.COLOR_BGR2GRAY)
        else:
            old_gray = self.processor.frames[0]  # Assuming it is already in grayscale
        
        points_tracked = {i: [p] for i, p in enumerate(self.processor.points_to_track)}
        
        for frame in self.processor.frames[1:]:
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame  # Assuming it is already in grayscale
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                points_tracked[i].append((a, b))
            
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        for i, frame in enumerate(self.processor.frames):
            for j, point in points_tracked.items():
                if i < len(point):
                    x, y = point[i]
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        self.processor.points_tracked = points_tracked
        self.display_clipped_frames()
        
        self.track_coords_button.config(state=tk.NORMAL)


    def plot_distances(self):
        if not hasattr(self, 'points_tracked'):
            return
        
        if len(self.axis_coords) < 5:
            print("Axes are not fully defined.")
            return
        
        origin = self.axis_coords[0]
        x_axis_end_pos = self.axis_coords[1]
        x_axis_end_neg = self.axis_coords[2]
        y_axis_end_pos = self.axis_coords[3]
        y_axis_end_neg = self.axis_coords[4]

        # Reference pixel distance and scale factor calculation
        ref_pixel_dist = np.linalg.norm(np.array(self.line_coords[0]) - np.array(self.line_coords[1]))
        scale_factor = self.ref_distance / ref_pixel_dist

        for i, points in self.points_tracked.items():
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
        end_index = min(len(self.processor.frames), middle_frame_index + 10 )
        
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
                    real_coords = self.translate_to_real_coordinates(self.processor.points_tracked[j])
                    x, y = real_coords[i]
                    row.append(f"({x:.2f}, {y:.2f})")
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



    def choose_tracking_method(self):
        self.tracking_method = tk.StringVar()
        self.tracking_method.set("points")  # default value

        method_popup = tk.Toplevel(self.root)
        method_popup.title("Select Tracking Method")

        tk.Label(method_popup, text="Choose tracking method:").pack(pady=10)
        tk.Radiobutton(method_popup, text="Track Points", variable=self.tracking_method, value="points").pack(anchor=tk.W)
        tk.Radiobutton(method_popup, text="Track Centroid", variable=self.tracking_method, value="bbox").pack(anchor=tk.W)

        tk.Button(method_popup, text="OK", command=lambda: self.confirm_tracking_method(method_popup)).pack(pady=10)

    def confirm_tracking_method(self, popup):
        method = self.tracking_method.get()
        popup.destroy()

        if method == "points":
            self.mark_points_to_track()
        elif method == "bbox":
            self.mark_bboxes_to_track()
        else:
            messagebox.showerror("Error", "Invalid method. Please select 'points' or 'bbox'.")

    def mark_bboxes_to_track(self):
        self.bboxes_to_track = []
        messagebox.showinfo("Instruction", "Please draw bounding boxes around the objects to track. Draw multiple bounding boxes if needed.")
        self.video_view.bind("<Button-1>", self.start_bbox)

    def start_bbox(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.current_bbox = self.video_view.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red")
        self.video_view.bind("<B1-Motion>", self.draw_bbox)
        self.video_view.bind("<ButtonRelease-1>", self.finish_bbox)

    def draw_bbox(self, event):
        self.video_view.coords(self.current_bbox, self.start_x, self.start_y, event.x, event.y)

    def finish_bbox(self, event):
        self.video_view.unbind("<B1-Motion>")
        self.video_view.unbind("<ButtonRelease-1>")
        bbox_coords = (self.start_x, self.start_y, event.x, event.y)
        self.bboxes_to_track.append(bbox_coords)
        centroid_x = (bbox_coords[0] + bbox_coords[2]) / 2
        centroid_y = (bbox_coords[1] + bbox_coords[3]) / 2
        self.processor.points_to_track.append((centroid_x, centroid_y))
        self.video_view.create_oval(centroid_x - 3, centroid_y - 3, centroid_x + 3, centroid_y + 3, fill="red")

    def show_tracked_coordinates_window(self):
        popup = tk.Toplevel(self.root)
        popup.title("Tracked Coordinates Options")

        tk.Button(popup, text="Plot X and Y Distances", command=self.plot_distances).pack(pady=5)
        tk.Button(popup, text="1st Derivative (Velocity)", command=lambda: self.calculate_and_export_derivative(1)).pack(pady=5)
        tk.Button(popup, text="2nd Derivative (Acceleration)", command=lambda: self.calculate_and_export_derivative(2)).pack(pady=5)
        # tk.Button(popup, text="Calculate Path Length", command=self.calculate_path_length).pack(pady=5)
        # tk.Button(popup, text="Calculate Angle of Movement", command=self.calculate_angle_of_movement).pack(pady=5)
        tk.Button(popup, text="Track Angle Between Three Points", command=self.select_third_point).pack(pady=5)
        tk.Button(popup, text="Show Tracked Points Table", command=self.show_tracked_points_table).pack(pady=5)

    def plot_distances(self):
        if not hasattr(self.processor, 'points_tracked'):
            messagebox.showerror("Error", "No tracked points available. Please start tracking first.")
            return
        
        self.processor.plot_distances()

    def calculate_and_export_derivative(self, order):
        if not hasattr(self.processor, 'points_tracked'):
            messagebox.showerror("Error", "No tracked points available. Please start tracking first.")
            return

        popup = tk.Toplevel(self.root)
        popup.title(f"{order}st Derivative Options")

        def plot_derivative():
            derivative = self.calculate_derivative(order)
            self.plot_derivative(derivative, order)

        def export_derivative():
            derivative = self.calculate_derivative(order)
            self.export_derivative_to_csv(derivative, order)

        tk.Button(popup, text="Plot", command=plot_derivative).pack(pady=5)
        tk.Button(popup, text="Export", command=export_derivative).pack(pady=5)

    def calculate_derivative(self, order):
        fps = self.processor.fps
        derivative = {}

        for key, points in self.processor.points_tracked.items():
            real_coords = self.translate_to_real_coordinates(points)
            positions = np.array(real_coords)
            if order == 1:
                derivative[key] = np.gradient(positions, axis=0) * fps
            elif order == 2:
                first_derivative = np.gradient(positions, axis=0) * fps
                derivative[key] = np.gradient(first_derivative, axis=0) * fps

        return derivative

    def plot_derivative(self, derivative, order):
        for key, values in derivative.items():
            x_values = [v[0] for v in values]
            y_values = [v[1] for v in values]

            plt.figure()
            plt.plot(x_values, label='X')
            plt.plot(y_values, label='Y')
            plt.title(f"{order}st Derivative for Point {key+1}")
            plt.xlabel("Frame")
            plt.ylabel(f"{order}st Derivative Value")
            plt.legend()
            plt.show()

    def export_derivative_to_csv(self, derivative, order):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            return

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ["Frame"] + [f"Point {i+1} (dx, dy)" for i in range(len(derivative))]
            writer.writerow(header)

            max_len = max(len(values) for values in derivative.values())
            for i in range(max_len):
                row = [i]
                for values in derivative.values():
                    if i < len(values):
                        row.append(f"({values[i][0]:.2f}, {values[i][1]:.2f})")
                    else:
                        row.append("")
                writer.writerow(row)

    def select_third_point(self):
        if len(self.processor.points_to_track) < 2:
            messagebox.showerror("Error", "Please mark at least two points to track first.")
            return

        if len(self.processor.points_to_track) > 2:
            self.show_point_selection_popup()
        else:
            self.selected_points = [0, 1]  # Automatically select the first two points
            self.third_point_selection_proceed()

    def show_point_selection_popup(self):
        self.point_selection_popup = tk.Toplevel(self.root)
        self.point_selection_popup.title("Select Two Points")

        tk.Label(self.point_selection_popup, text="Select two points:").pack(pady=10)

        self.point_vars = []
        for i, point in enumerate(self.processor.points_to_track):
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.point_selection_popup, text=f"Point {i+1}", variable=var)
            chk.pack(anchor=tk.W)
            self.point_vars.append(var)

        tk.Button(self.point_selection_popup, text="Confirm", command=self.confirm_selected_points).pack(pady=10)

    def confirm_selected_points(self):
        selected_points = [i for i, var in enumerate(self.point_vars) if var.get()]
        if len(selected_points) != 2:
            messagebox.showerror("Error", "Please select exactly two points.")
            return
        self.selected_points = selected_points
        self.point_selection_popup.destroy()
        self.third_point_selection_proceed()

    def third_point_selection_proceed(self):
        messagebox.showinfo("Instruction", "Please select the third point to track the angle.")
        self.video_view.bind("<Button-1>", self.mark_third_point)

    def mark_third_point(self, event):
        self.processor.third_point = (event.x, event.y)
        self.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="blue")
        self.video_view.unbind("<Button-1>")
        self.track_angle_between_points()

    def track_angle_between_points(self):
        if not hasattr(self.processor, 'third_point'):
            messagebox.showerror("Error", "Please mark the third point to track the angle.")
            return

        # Ensure the third point is added for all frames
        p3 = np.array([self.processor.third_point for _ in range(len(self.processor.frames))], dtype=np.float32)

        # Select the two tracked points chosen by the user
        p1 = np.array(self.processor.points_tracked[self.selected_points[0]], dtype=np.float32)
        p2 = np.array(self.processor.points_tracked[self.selected_points[1]], dtype=np.float32)

        angles_over_time = []

        for i in range(len(self.processor.frames)):
            if i < len(p1) and i < len(p2):
                a = p1[i]
                b = p2[i]
                c = p3[i]

                angle = self.calculate_angle(a, b, c)
                angles_over_time.append(angle)

        self.processor.angles_over_time = angles_over_time

        # Enable visualization and export options
        popup = tk.Toplevel(self.root)
        popup.title("Angle Tracking Options")

        tk.Button(popup, text="Visualize Angles", command=self.visualize_angles).pack(pady=5)
        tk.Button(popup, text="Export Angles to CSV", command=self.export_angles_to_csv).pack(pady=5)

    def calculate_angle(self, a, b, c):
        # Calculate the angle between three points (a, b, c)
        ba = a - b
        bc = c - b
        ba = ba.reshape(-1)  # Reshape to 1D array
        bc = bc.reshape(-1)  # Reshape to 1D array
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def visualize_angles(self):
        if not hasattr(self.processor, 'angles_over_time'):
            messagebox.showerror("Error", "No angle data available. Please track the angles first.")
            return

        plt.figure()
        plt.plot(self.processor.angles_over_time, label='Angle')
        plt.title("Angle Between Three Points Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Angle (degrees)")
        plt.legend()
        plt.show()

    def export_angles_to_csv(self):
        if not hasattr(self.processor, 'angles_over_time'):
            messagebox.showerror("Error", "No angle data available. Please track the angles first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file_path:
            return

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Angle (degrees)"])

            for i, angle in enumerate(self.processor.angles_over_time):
                writer.writerow([i, angle])

    def back_to_menu(self):
        self.root.destroy()
        root = tk.Tk()
        root.geometry("960x640")
        menu_x = MenuScreen(root)
        root.mainloop()

    def on_close(self):
        self.root.destroy()