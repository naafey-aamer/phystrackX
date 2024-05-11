import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from VideoObjectTracker import VideoObjectTracker

        
def main():
    root = tk.Tk()
    app = VideoObjectTracker(root)
    root.mainloop()

if __name__ == "__main__":
    main()
