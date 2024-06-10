import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import ImageTk, Image, ImageDraw
from video_processing import VideoProcessor
from utils import resize_frame
import cv2
import numpy as np
from tkinter import ttk

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phys Tracker")
        
        # Make the window cover the entire screen
        self.root.attributes('-fullscreen', True)
        
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
        
        self.undo_button = tk.Button(self.filter_frame, text="Undo Filters", command=self.undo_filter)
        self.undo_button.pack(pady=10)
        
        self.frame_button = tk.Button(self.filter_frame, text="Select Initial and Final Frames", command=self.select_frames)
        self.frame_button.pack(pady=10)
        
        self.distance_button = tk.Button(self.filter_frame, text="Set Reference Distance", command=self.set_reference_distance)
        self.distance_button.pack(pady=10)
        
        self.clip_button = tk.Button(self.filter_frame, text="Clip Video", command=self.clip_video)
        self.clip_button.pack(pady=10)
        
        self.axis_button = tk.Button(self.filter_frame, text="Mark Axes", command=self.mark_axes)
        self.axis_button.pack(pady=10)
        
        self.track_button = tk.Button(self.filter_frame, text="Mark Points to Track", command=self.mark_points_to_track)
        self.track_button.pack(pady=10)
        
        self.track_start_button = tk.Button(self.filter_frame, text="Start Tracking", command=self.start_tracking)
        self.track_start_button.pack(pady=10)
        
        self.plot_button = tk.Button(self.filter_frame, text="Plot X and Y Distances", command=self.plot_distances)
        self.plot_button.pack(pady=10)
        self.plot_button.config(state=tk.DISABLED)
        
        self.info_label = tk.Label(self.filter_frame, text="")
        self.info_label.pack(pady=10)
        
        self.exit_fullscreen_button = tk.Button(self.filter_frame, text="Exit Fullscreen", command=self.exit_fullscreen)
        self.exit_fullscreen_button.pack(pady=10)
        
        # Create a frame to hold the video and slider widgets on the right side
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_view = tk.Canvas(self.video_frame, width=640, height=480)
        self.video_view.pack(pady=20, expand=True)
        
        self.slider = tk.Scale(self.video_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=400, resolution=1, command=self.update_frame)
        self.slider.pack(pady=10)

    
    def open_video(self):
        self.processor.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.MP4")])
        if self.processor.video_path:
            self.processor.fps = int(simpledialog.askinteger("FPS", "Enter FPS:"))
            self.fps_label.config(text=f"FPS: {self.processor.fps}")
            self.load_video()
    
    def load_video(self):
        cap = cv2.VideoCapture(self.processor.video_path)
        self.processor.frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = resize_frame(frame, 640, 480)
            self.processor.frames.append(frame)

        if self.processor.frames:
            frame = self.processor.frames[0]
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=img)
            self.video_view.create_image(0, 0, image=photo, anchor='nw')
            self.photo = photo

            self.slider['to'] = len(self.processor.frames) - 1
        cap.release()


    def update_frame(self, event):
        frame_number = int(self.slider.get())
        if self.processor.filtered_images:
            frame = self.processor.filtered_images[frame_number]
        elif self.processor.cropped_frames:
            frame = self.processor.cropped_frames[frame_number]
        else:
            cap = cv2.VideoCapture(self.processor.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                return
            frame = resize_frame(frame, 640, 480)
            frame = np.uint8(np.clip(frame, 0, 255))
            cap.release()

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=img)

        self.video_view.delete('all')
        self.video_view.create_image(0, 0, image=photo, anchor='nw')
        self.photo = photo



    def select_frames(self):
        self.processor.initial_frame = int(simpledialog.askinteger("Initial Frame", "Enter the initial frame number:"))
        self.processor.final_frame = int(simpledialog.askinteger("Final Frame", "Enter the final frame number:"))
        self.info_label.config(text=f"Initial Frame: {self.processor.initial_frame}, Final Frame: {self.processor.final_frame}")
    
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
        self.processor.frames = self.processor.cropped_frames
        self.processor.cropped_frames = []

        # If there are filtered images, apply the filters to the clipped frames
        if self.processor.filtered_images:
            self.processor.apply_filters_to_clipped_frames()

        self.display_clipped_frames()

    def display_clipped_frames(self):
        def update_display(frame_index):
            if self.processor.filtered_images:
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
        self.processor.axis_coords = [(event.x, event.y)]
        self.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="blue")
        self.video_view.unbind("<Button-1>")
        messagebox.showinfo("Instruction", "Now, move the cursor to draw the axes and click to drop the end points.")
        self.video_view.bind("<Motion>", self.update_axes_image)
        self.video_view.bind("<Button-1>", self.drop_axes)
    
    def update_axes_image(self, event):
        if len(self.processor.axis_coords) == 1:
            origin = self.processor.axis_coords[0]
            if self.processor.axis_image_id:
                self.video_view.delete(self.processor.axis_image_id)
            img = Image.new('RGBA', (640, 480), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.line([origin, (event.x, origin[1])], fill="blue", width=2)  # X-axis
            draw.line([origin, (origin[0], event.y)], fill="green", width=2)  # Y-axis
            self.processor.axis_image = ImageTk.PhotoImage(img)
            self.processor.axis_image_id = self.video_view.create_image(0, 0, image=self.processor.axis_image, anchor='nw')
    
    def drop_axes(self, event):
        if len(self.processor.axis_coords) == 1:
            origin = self.processor.axis_coords[0]
            self.processor.axis_coords.append((event.x, origin[1]))  # X-axis end point
            self.processor.axis_coords.append((origin[0], event.y))  # Y-axis end point
            self.video_view.create_line(origin[0], origin[1], event.x, origin[1], fill="blue", width=2)  # X-axis
            self.video_view.create_line(origin[0], origin[1], origin[0], event.y, fill="green", width=2)  # Y-axis
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
        if not self.processor.frames:
            messagebox.showerror("Error", "Please clip the video first.")
            return
        
        if not self.processor.points_to_track:
            messagebox.showerror("Error", "Please mark points to track first.")
            return
        
        p0 = np.array(self.processor.points_to_track, dtype=np.float32).reshape(-1, 1, 2)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        old_gray = cv2.cvtColor(self.processor.frames[0], cv2.COLOR_BGR2GRAY)
        points_tracked = {i: [p] for i, p in enumerate(self.processor.points_to_track)}
        
        for frame in self.processor.frames[1:]:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        
        self.plot_button.config(state=tk.NORMAL)

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
            "Band-pass Filter",
            "Bilateral Filter",
            "Object Separation",
            "Contrast Adjustment"
        ]
        
        self.filter_vars = {filter_name: tk.BooleanVar() for filter_name in filters}
        
        for filter_name in filters:
            tk.Checkbutton(popup, text=filter_name, variable=self.filter_vars[filter_name]).pack(anchor='w')
        
        apply_button = tk.Button(popup, text="Apply", command=lambda: self.apply_filters(popup))
        apply_button.pack(pady=10)
    
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
        self.update_frame(None)
        self.display_clipped_frames()

    def undo_filter(self):
        self.processor.filtered_images = []
        self.update_frame(None)
    
    def on_close(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
