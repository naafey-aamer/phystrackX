import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from scipy.signal import butter, filtfilt
from utils import resize_frame

class VideoProcessor:
    def __init__(self):
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

        self.prev_smoothed_frame = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.abg_state = np.zeros((2,))
        self.abg_velocity = np.zeros((2,))
        self.abg_acceleration = np.zeros((2,))

    def clip_video(self):
            if self.initial_frame is None or self.final_frame is None:
                raise ValueError("Initial and final frames must be set before clipping the video.")
            
            self.cropped_frames = self.frames[self.initial_frame:self.final_frame + 1]
            
            # Optionally, update self.filtered_images based on the cropped frames if needed
            if self.filtered_images:
                self.filtered_images = self.filtered_images[self.initial_frame:self.final_frame + 1]


    def apply_filters(self, selected_filters):
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

    def apply_filters_to_clipped_frames(self):
        if not self.cropped_frames or not self.filtered_images:
            return

        for i, frame in enumerate(self.cropped_frames):
            for filter_type in self.filter_vars.keys():
                if self.filter_vars[filter_type].get():
                    frame = self.apply_filter_to_frame(filter_type, frame)
            self.cropped_frames[i] = frame

    def apply_exponential_smoothing(self, frame, alpha=0.1):
        if self.prev_smoothed_frame is None:
            self.prev_smoothed_frame = frame
        smoothed_frame = alpha * frame + (1 - alpha) * self.prev_smoothed_frame
        self.prev_smoothed_frame = smoothed_frame
        return smoothed_frame

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=0)
        return y

    def apply_filter_to_frame(self, frame, filter_type, prev_frame=None):
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
            fg_mask_3channel = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, fg_mask_3channel)
        
        elif filter_type == "Low-pass Filter":
            frame = cv2.GaussianBlur(frame, (15, 15), 0)

        elif filter_type == "High-pass Filter":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            high_pass = cv2.Laplacian(gray, cv2.CV_64F)
            frame = cv2.convertScaleAbs(high_pass)

        elif filter_type == "Median Filter":
            frame = cv2.medianBlur(frame, 5)

        elif filter_type == "Exponential Smoothing":
            frame = self.apply_exponential_smoothing(frame)
            frame = np.uint8(np.clip(frame, 0, 255))

        elif filter_type == "Band-pass Filter":
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.Laplacian(frame, cv2.CV_64F)
            frame = np.uint8(np.absolute(frame))

        elif filter_type == "Bilateral Filter":
            frame = cv2.bilateralFilter(frame, 9, 75, 75)

        elif filter_type == "Object Separation":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output = np.zeros_like(frame)
            for i, contour in enumerate(contours):
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.drawContours(output, contours, i, color, -1)
            frame = output

        elif filter_type == "Contrast Adjustment":
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(img)
            frame = np.array(enhancer.enhance(1))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        elif filter_type == "Optical Flow":
            if prev_frame is None:
                return frame
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=1, winsize=35,
                iterations=3, poly_n=7, poly_sigma=3.0, flags=0
            )
            
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame

    def start_tracking(self):
        if not self.cropped_frames:
            return
        
        if not self.points_to_track:
            return
        
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
        
        for i, frame in enumerate(self.cropped_frames):
            for j, point in points_tracked.items():
                if i < len(point):
                    x, y = point[i]
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        self.points_tracked = points_tracked

    def plot_distances(self):
        if not hasattr(self, 'points_tracked'):
            return
        
        origin = self.axis_coords[0]
        x_axis_end = self.axis_coords[1]
        y_axis_end = self.axis_coords[2]
        
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