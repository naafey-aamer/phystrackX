import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import ImageTk, Image, ImageDraw, ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# from concurrent.futures import ProcessPoolExecutor, as_completed


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
        self.filtered_images = []

        # Initialize exponential smoothing state        
        self.prev_smoothed_frame = None

        # Initialize alpha-beta-gamma filter state
        self.abg_state = np.zeros((2,))
        self.abg_velocity = np.zeros((2,))
        self.abg_acceleration = np.zeros((2,))

        # Initialize GMM background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        # Create widgets
        self.create_widgets()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_widgets(self):
        # Button to open file dialog
        self.open_button = tk.Button(self.root, text="Open Video", command=self.open_video)
        self.open_button.pack(pady=10)
        
        # Label to display fps
        self.fps_label = tk.Label(self.root, text="")
        self.fps_label.pack(pady=5)
        
        # Button for filter options
        self.filter_button = tk.Button(self.root, text="Apply Filters", command=self.show_filter_popup)
        self.filter_button.pack(pady=10)
        
        # Button to undo filters
        self.undo_button = tk.Button(self.root, text="Undo Filters", command=self.undo_filter)
        self.undo_button.pack(pady=10)
        
        # Video view
        self.video_view = tk.Canvas(self.root, width=640, height=480)
        self.video_view.pack(pady=20)
        
        # Slider for selecting frames
        self.slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, length=400, resolution=1, command=self.update_frame)
        self.slider.pack(pady=10)
        
        # Button to select initial and final frames
        self.frame_button = tk.Button(self.root, text="Select Initial and Final Frames", command=self.select_frames)
        self.frame_button.pack(pady=10)
        
        # Button to set reference distance
        self.distance_button = tk.Button(self.root, text="Set Reference Distance", command=self.set_reference_distance)
        self.distance_button.pack(pady=10)
        
        # Button to clip video
        self.clip_button = tk.Button(self.root, text="Clip Video", command=self.clip_video)
        self.clip_button.pack(pady=10)
        
        # Button to mark axes
        self.axis_button = tk.Button(self.root, text="Mark Axes", command=self.mark_axes)
        self.axis_button.pack(pady=10)
        
        # Button to mark points to track
        self.track_button = tk.Button(self.root, text="Mark Points to Track", command=self.mark_points_to_track)
        self.track_button.pack(pady=10)
        
        # Button to start tracking
        self.track_start_button = tk.Button(self.root, text="Start Tracking", command=self.start_tracking)
        self.track_start_button.pack(pady=10)
        
        # Button to plot x and y distances
        self.plot_button = tk.Button(self.root, text="Plot X and Y Distances", command=self.plot_distances)
        self.plot_button.pack(pady=10)
        self.plot_button.config(state=tk.DISABLED)
        
        # Label to display selected frames and reference distance
        self.info_label = tk.Label(self.root, text="")
        self.info_label.pack(pady=10)
    
    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.MP4")])
        if self.video_path:
            self.fps = int(simpledialog.askinteger("FPS", "Enter FPS:"))
            self.fps_label.config(text=f"FPS: {self.fps}")
            self.load_video()
    
    def resize_frame(self, frame, width, height):
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Error reading video")
            return

        # Resize frame to fit within the canvas
        frame = self.resize_frame(frame, 640, 480)

        # Convert frame to PhotoImage for display
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=img)

        # Display the first frame
        self.video_view.create_image(0, 0, image=photo, anchor='nw')
        self.photo = photo

        # Update slider range based on video duration
        self.slider['to'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()

    def update_frame(self, event):
        frame_number = int(self.slider.get())
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            return

        # Apply filters if any
        if self.filtered_images:
            frame = self.filtered_images[frame_number]

        # Resize frame to fit within the canvas
        frame = self.resize_frame(frame, 640, 480)

        # Ensure frame is in uint8 format
        frame = np.uint8(np.clip(frame, 0, 255))

        # Convert frame to PhotoImage for display
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=img)

        # Update canvas with new frame
        self.video_view.delete('all')
        self.video_view.create_image(0, 0, image=photo, anchor='nw')
        self.photo = photo
        cap.release()

    
    def select_frames(self):
        self.initial_frame = int(simpledialog.askinteger("Initial Frame", "Enter the initial frame number:"))
        self.final_frame = int(simpledialog.askinteger("Final Frame", "Enter the final frame number:"))
        self.info_label.config(text=f"Initial Frame: {self.initial_frame}, Final Frame: {self.final_frame}")
    
    def set_reference_distance(self):
        self.line_coords = []
        messagebox.showinfo("Instruction", "Please click two points on the video to set the reference distance.")
        self.video_view.bind("<Button-1>", self.mark_line)
    
    def mark_line(self, event):
        if len(self.line_coords) < 2:
            self.line_coords.append((event.x, event.y))
            if len(self.line_coords) == 2:
                self.video_view.create_line(self.line_coords[0], self.line_coords[1], fill="red", width=2)
                self.ref_distance = simpledialog.askfloat("Reference Distance", "Enter the reference distance in your chosen unit:")
                self.info_label.config(text=f"Reference Distance: {self.ref_distance} units")
    
    def clip_video(self):
        if self.initial_frame is None or self.final_frame is None:
            messagebox.showerror("Error", "Please select initial and final frames first.")
            return

        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cropped_frames = []

        for frame_num in range(self.initial_frame, self.final_frame + 1):
            if frame_num >= frame_count:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.resize_frame(frame, 640, 480)  # Resize frame to fit within the canvas
            self.cropped_frames.append(frame)

        cap.release()
        
        if self.filtered_images:
            self.apply_filters_to_clipped_frames()

        self.display_clipped_frames()
        self.start_tracking()
        
    def display_clipped_frames(self):
        if not self.cropped_frames:
            return

        def update_display(frame_index):
            frame = self.cropped_frames[frame_index]
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=img)
            self.video_view.delete('all')
            self.video_view.create_image(0, 0, image=photo, anchor='nw')
            self.photo = photo

        self.slider['to'] = len(self.cropped_frames) - 1
        self.slider.config(command=lambda event: update_display(int(self.slider.get())))
        update_display(0)

    
    def mark_axes(self):
        self.axis_coords = []
        messagebox.showinfo("Instruction", "Please click to set the origin of the axes.")
        self.video_view.bind("<Button-1>", self.set_origin)
    
    def set_origin(self, event):
        self.axis_coords = [(event.x, event.y)]
        self.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="blue")
        self.video_view.unbind("<Button-1>")
        messagebox.showinfo("Instruction", "Now, move the cursor to draw the axes and click to drop the end points.")
        self.video_view.bind("<Motion>", self.update_axes_image)
        self.video_view.bind("<Button-1>", self.drop_axes)
    
    def update_axes_image(self, event):
        if len(self.axis_coords) == 1:
            origin = self.axis_coords[0]
            if self.axis_image_id:
                self.video_view.delete(self.axis_image_id)
            img = Image.new('RGBA', (640, 480), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.line([origin, (event.x, origin[1])], fill="blue", width=2)  # X-axis
            draw.line([origin, (origin[0], event.y)], fill="green", width=2)  # Y-axis
            self.axis_image = ImageTk.PhotoImage(img)
            self.axis_image_id = self.video_view.create_image(0, 0, image=self.axis_image, anchor='nw')
    
    def drop_axes(self, event):
        if len(self.axis_coords) == 1:
            origin = self.axis_coords[0]
            self.axis_coords.append((event.x, origin[1]))  # X-axis end point
            self.axis_coords.append((origin[0], event.y))  # Y-axis end point
            self.video_view.create_line(origin[0], origin[1], event.x, origin[1], fill="blue", width=2)  # X-axis
            self.video_view.create_line(origin[0], origin[1], origin[0], event.y, fill="green", width=2)  # Y-axis
            self.video_view.unbind("<Motion>")
            self.video_view.unbind("<Button-1>")
    
    def mark_points_to_track(self):
        self.points_to_track = []
        messagebox.showinfo("Instruction", "Please click points on the video to mark them for tracking.")
        self.video_view.bind("<Button-1>", self.mark_point)
    
    def mark_point(self, event):
        self.points_to_track.append((event.x, event.y))
        self.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
    
    def start_tracking(self):
        if not self.cropped_frames:
            messagebox.showerror("Error", "Please clip the video first.")
            return
        
        if not self.points_to_track:
            messagebox.showerror("Error", "Please mark points to track first.")
            return
        
        # Prepare tracking points
        p0 = np.array(self.points_to_track, dtype=np.float32).reshape(-1, 1, 2)
        
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        old_gray = cv2.cvtColor(self.cropped_frames[0], cv2.COLOR_BGR2GRAY)
        points_tracked = {i: [p] for i, p in enumerate(self.points_to_track)}
        
        for frame in self.cropped_frames[1:]:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                points_tracked[i].append((a, b))
            
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        # Display tracked points on each frame
        for i, frame in enumerate(self.cropped_frames):
            for j, point in points_tracked.items():
                if i < len(point):
                    x, y = point[i]
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        self.display_clipped_frames()
        
        # Output the tracked points
        for i, points in points_tracked.items():
            print(f"Point {i}:")
            for point in points:
                print(f"    {point}")
        
        # Store tracked points for plotting
        self.points_tracked = points_tracked
        
        # Enable the plot button after tracking
        self.plot_button.config(state=tk.NORMAL)
    
    def plot_distances(self):
        if not hasattr(self, 'points_tracked'):
            messagebox.showerror("Error", "No tracked points available. Please start tracking first.")
            return
        
        origin = self.axis_coords[0]
        x_axis_end = self.axis_coords[1]
        y_axis_end = self.axis_coords[2]
        
        # Compute the scale factor
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

        self.filtered_images = []
        cap = cv2.VideoCapture(self.video_path)
        prev_frame = None
        
        for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break

            for filter_type in selected_filters:
                frame = self.apply_filter_to_frame(filter_type, frame, prev_frame)
            
            self.filtered_images.append(frame)
            prev_frame = frame

        cap.release()
        popup.destroy()
        self.update_frame(None)
        self.apply_filters_to_clipped_frames()

    def apply_filters_to_clipped_frames(self):
        if not self.cropped_frames or not self.filtered_images:
            return
        
        for i, frame in enumerate(self.cropped_frames):
            for filter_type in self.filter_vars.keys():
                if self.filter_vars[filter_type].get():
                    frame = self.apply_filter_to_frame(filter_type, frame)
            self.cropped_frames[i] = frame

    def apply_exponential_smoothing(self, frame, alpha=0.1):
        global prev_smoothed_frame
        if self.prev_smoothed_frame is None:
            self.prev_smoothed_frame = frame
        smoothed_frame = alpha * frame + (1 - alpha) * self.prev_smoothed_frame
        self.prev_smoothed_frame = smoothed_frame
        return smoothed_frame

    def apply_abg_filter(self, frame, alpha=0.85, beta=0.005, gamma=0.001):
        # self.initialize_abg_filter()
        measurement = np.array([np.mean(frame[:, :, 0]), np.mean(frame[:, :, 1])])  # Simplified for demonstration
        self.abg_state += self.abg_velocity + 0.5 * self.abg_acceleration
        self.abg_velocity += self.abg_acceleration
        residual = measurement - self.abg_state
        self.abg_state += alpha * residual
        self.abg_velocity += beta * residual
        self.abg_acceleration += gamma * residual
        return np.uint8(self.abg_state).reshape((1, 1, 2))  # Simplified to a single pixel for demonstration

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=0)
        return y

    def apply_filter_to_frame(self, filter_type, frame, prev_frame=None):
        global mean_shift_initialized, camshift_initialized
        
        img = None  # Ensure img is always initialized

        if filter_type == "GMM With Background":
                fg_mask = self.bg_subtractor.apply(frame)
                _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        elif filter_type == "GMM Without Background":
            fg_mask = self.bg_subtractor.apply(frame)
            _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
            # Mask the original frame with the foreground mask
            fg_mask_3channel = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, fg_mask_3channel)
        
        elif filter_type == "Mean Shift Filter":
            pass

        elif filter_type == "Camshift":
            pass

        elif filter_type == "Low-pass Filter":
            # Apply Gaussian blur as a low-pass filter
            frame = cv2.GaussianBlur(frame, (15, 15), 0)

        elif filter_type == "High-pass Filter":
            # Apply high-pass filter using Laplacian
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            high_pass = cv2.Laplacian(gray, cv2.CV_64F)
            frame = cv2.convertScaleAbs(high_pass)

        elif filter_type == "Median Filter":
            # Apply median filter
            frame = cv2.medianBlur(frame, 5)

        elif filter_type == "Exponential Smoothing":
            frame = self.apply_exponential_smoothing(frame)
            frame = np.uint8(np.clip(frame, 0, 255))  # Ensure frame is in uint8 format

        elif filter_type == "Alpha-Beta-Gamma Filter":
            frame = self.apply_abg_filter(frame)
            frame = np.uint8(np.clip(frame, 0, 255))  # Ensure frame is in uint8 format

        elif filter_type == "Band-pass Filter":
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.Laplacian(frame, cv2.CV_64F)
            frame = np.uint8(np.absolute(frame))

        elif filter_type == "Bilateral Filter":
            # Apply bilateral filter
            frame = cv2.bilateralFilter(frame, 9, 75, 75)

        elif filter_type == "Object Separation":
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contours with different colors
            output = np.zeros_like(frame)
            for i, contour in enumerate(contours):
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.drawContours(output, contours, i, color, -1)
            frame = output

        elif filter_type == "Contrast Adjustment":
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(img)
            frame = np.array(enhancer.enhance(2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        elif filter_type == "Optical Flow":
            if prev_frame is None:
                return frame
            
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculate optical flow with less sensitivity
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=1, winsize=35,
                iterations=3, poly_n=7, poly_sigma=3.0, flags=0
            )
            
            # Convert flow to color image
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame

    
    def undo_filter(self):
        self.filtered_images = []
        self.update_frame(None)
    
    def on_close(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()