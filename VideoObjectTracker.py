from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tkinter as tk

class VideoObjectTracker:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Object Tracker")
        
        self.video_path = ""
        self.initial_frame = None
        self.final_frame = None
        self.object_points = None
        self.reference_point = None
        self.origin_point = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Button to upload video
        self.upload_button = tk.Button(self.master, text="Upload Video", command=self.upload_video)
        self.upload_button.pack()
        
        # Button to select initial and final frames
        self.frame_button = tk.Button(self.master, text="Select Frames", command=self.select_frames)
        self.frame_button.pack()
        
        # Button to mark object
        self.mark_button = tk.Button(self.master, text="Mark Object", command=self.mark_object)
        self.mark_button.pack()
        
        # Button to track object
        self.track_button = tk.Button(self.master, text="Track Object", command=self.track_object)
        self.track_button.pack()
        
    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if self.video_path:
            messagebox.showinfo("Video Uploaded", "Video has been uploaded successfully!")
    
    def select_frames(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video first!")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a frame selection dialog
        self.frame_dialog = tk.Toplevel(self.master)
        self.frame_dialog.title("Frame Selection")
        
        # Initial frame entry
        self.initial_frame_label = tk.Label(self.frame_dialog, text="Initial Frame:")
        self.initial_frame_label.grid(row=0, column=0)
        self.initial_frame_entry = tk.Entry(self.frame_dialog)
        self.initial_frame_entry.grid(row=0, column=1)
        
        # Final frame entry
        self.final_frame_label = tk.Label(self.frame_dialog, text="Final Frame:")
        self.final_frame_label.grid(row=1, column=0)
        self.final_frame_entry = tk.Entry(self.frame_dialog)
        self.final_frame_entry.grid(row=1, column=1)
        
        # Button to confirm frame selection
        self.confirm_button = tk.Button(self.frame_dialog, text="Confirm", command=self.confirm_frames)
        self.confirm_button.grid(row=2, columnspan=2)
        
    def confirm_frames(self):
        try:
            initial_frame = int(self.initial_frame_entry.get())
            final_frame = int(self.final_frame_entry.get())
            if initial_frame >= final_frame or initial_frame < 0 or final_frame < 0:
                messagebox.showerror("Error", "Invalid frame selection!")
                return
            self.initial_frame = initial_frame
            self.final_frame = final_frame
            self.frame_dialog.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid frame numbers!")
    
    def mark_object(self):
        if not all([self.video_path, self.initial_frame, self.final_frame]):
            messagebox.showerror("Error", "Please complete frame selection first!")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.initial_frame)
        
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Error reading video frame!")
            return
        
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mark feature points on the object
        object_points = cv2.goodFeaturesToTrack(frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        
        # Convert points to integers
        object_points = np.int0(object_points)
        
        # Draw circles around feature points
        for point in object_points:
            x, y = point.ravel()
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Display frame with feature points
        cv2.imshow("Mark Object", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        self.object_points = object_points
        
    def track_object(self):
        if not all([self.video_path, self.initial_frame, self.final_frame]):
            messagebox.showerror("Error", "Please complete frame selection and object marking first!")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.initial_frame)
        
        # Read the initial frame and convert it to grayscale
        ret, old_frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Error reading video frame!")
            return
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        # Define KLT parameters
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Select feature points for tracking
        object_points = self.object_points.astype(np.float32)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, object_points, None, **lk_params)
            
            # Select good points
            good_new = new_points[st == 1]
            good_old = object_points[st == 1]
            
            # Draw circles around feature points
            for point in good_new:
                x, y = point.ravel()
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                # cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Update the old frame and points
            old_gray = frame_gray.copy()
            object_points = good_new.reshape(-1, 1, 2)
            
            # Display the frame
            cv2.imshow("Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()