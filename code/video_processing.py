from tkinter import filedialog, simpledialog, messagebox
from PIL import ImageTk, Image, ImageDraw
import cv2
# from tracking import start_tracking

def open_video(app):
    app.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.MP4")])
    if app.video_path:
        app.fps = int(simpledialog.askinteger("FPS", "Enter FPS:"))
        app.fps_label.config(text=f"FPS: {app.fps}")
        load_video(app)

def load_video(app):
    cap = cv2.VideoCapture(app.video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image=img)
    
    app.video_view.create_image(0, 0, image=photo, anchor='nw')
    app.photo = photo
    
    app.slider['to'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    cap.release()

def update_frame(app, event):
    frame_number = int(app.slider.get())
    cap = cv2.VideoCapture(app.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        return
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image=img)
    
    app.video_view.delete('all')
    app.video_view.create_image(0, 0, image=photo, anchor='nw')
    app.photo = photo
    cap.release()

def select_frames(app):
    app.initial_frame = int(simpledialog.askinteger("Initial Frame", "Enter the initial frame number:"))
    app.final_frame = int(simpledialog.askinteger("Final Frame", "Enter the final frame number:"))
    app.info_label.config(text=f"Initial Frame: {app.initial_frame}, Final Frame: {app.final_frame}")

def set_reference_distance(app):
    app.line_coords = []
    messagebox.showinfo("Instruction", "Please click two points on the video to set the reference distance.")
    app.video_view.bind("<Button-1>", lambda event: mark_line(app, event))

def mark_line(app, event):
    if len(app.line_coords) < 2:
        app.line_coords.append((event.x, event.y))
        if len(app.line_coords) == 2:
            app.video_view.create_line(app.line_coords[0], app.line_coords[1], fill="red", width=2)
            app.ref_distance = simpledialog.askfloat("Reference Distance", "Enter the reference distance in your chosen unit:")
            app.info_label.config(text=f"Reference Distance: {app.ref_distance} units")

def clip_video(app):
    if app.initial_frame is None or app.final_frame is None:
        messagebox.showerror("Error", "Please select initial and final frames first.")
        return
    
    cap = cv2.VideoCapture(app.video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    app.cropped_frames = []

    for frame_num in range(app.initial_frame, app.final_frame + 1):
        if frame_num >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        app.cropped_frames.append(frame)
    
    cap.release()
    
    display_clipped_frames(app)
    start_tracking(app)

def display_clipped_frames(app):
    if not app.cropped_frames:
        return
    
    def update_display(frame_index):
        frame = app.cropped_frames[frame_index]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=img)
        app.video_view.delete('all')
        app.video_view.create_image(0, 0, image=photo, anchor='nw')
        app.photo = photo
    
    app.slider['to'] = len(app.cropped_frames) - 1
    app.slider.config(command=lambda event: update_display(int(app.slider.get())))
    update_display(0)

def mark_axes(app):
    app.axis_coords = []
    messagebox.showinfo("Instruction", "Please click to set the origin of the axes.")
    app.video_view.bind("<Button-1>", lambda event: set_origin(app, event))

def set_origin(app, event):
    app.axis_coords = [(event.x, event.y)]
    app.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="blue")
    app.video_view.unbind("<Button-1>")
    messagebox.showinfo("Instruction", "Now, move the cursor to draw the axes and click to drop the end points.")
    app.video_view.bind("<Motion>", lambda event: update_axes_image(app, event))
    app.video_view.bind("<Button-1>", lambda event: drop_axes(app, event))

def update_axes_image(app, event):
    if len(app.axis_coords) == 1:
        origin = app.axis_coords[0]
        if app.axis_image_id:
            app.video_view.delete(app.axis_image_id)
        img = Image.new('RGBA', (640, 480), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.line([origin, (event.x, origin[1])], fill="blue", width=2)  # X-axis
        draw.line([origin, (origin[0], event.y)], fill="green", width=2)  # Y-axis
        app.axis_image = ImageTk.PhotoImage(img)
        app.axis_image_id = app.video_view.create_image(0, 0, image=app.axis_image, anchor='nw')

def drop_axes(app, event):
    if len(app.axis_coords) == 1:
        origin = app.axis_coords[0]
        app.axis_coords.append((event.x, origin[1]))  # X-axis end point
        app.axis_coords.append((origin[0], event.y))  # Y-axis end point
        app.video_view.create_line(origin[0], origin[1], event.x, origin[1], fill="blue", width=2)  # X-axis
        app.video_view.create_line(origin[0], origin[1], origin[0], event.y, fill="green", width=2)  # Y-axis
        app.video_view.unbind("<Motion>")
        app.video_view.unbind("<Button-1>")

def mark_points_to_track(app):
    app.points_to_track = []
    messagebox.showinfo("Instruction", "Please click points on the video to mark them for tracking.")
    app.video_view.bind("<Button-1>", lambda event: mark_point(app, event))

def mark_point(app, event):
    app.points_to_track.append((event.x, event.y))
    app.video_view.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
