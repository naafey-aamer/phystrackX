import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import ImageTk, Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.MP4" )])
        if self.video_path:
            self.fps = int(simpledialog.askinteger("FPS", "Enter FPS:"))
            self.fps_label.config(text=f"FPS: {self.fps}")
            self.load_video()
    
    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Error reading video")
            return
        
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
            self.cropped_frames.append(frame)
        
        cap.release()
        
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
    
    def on_close(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
