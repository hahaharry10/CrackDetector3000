import tkinter as tk
from tkinter import filedialog

class CrackDetector3000(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set title
        self.title("CrackDetector3000")

        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set dynamic window size (70% of screen resolution)
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.7)

        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Make window resizable
        self.columnconfigure(0, weight=1)
        self.rowconfigure([0, 1, 2, 3], weight=1)

        # Create title label
        self.label = tk.Label(self, text="Welcome to CrackDetector3000", font=("Arial", 20, "bold"))
        self.label.grid(row=0, column=0, pady=20, sticky="nsew")

        # Create buttons
        self.generate_map_btn = tk.Button(self, text="Generate Map", font=("Arial", 14), command=self.generate_map)
        self.generate_map_btn.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        self.detect_anomaly_btn = tk.Button(self, text="Detect Anomaly", font=("Arial", 14), command=self.detect_anomaly)
        self.detect_anomaly_btn.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        self.manual_input_btn = tk.Button(self, text="Manual Input", font=("Arial", 14), command=self.open_manual_input)
        self.manual_input_btn.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

    def generate_map(self):
        """Function triggered when 'Generate Map' is clicked."""
        self.label.config(text="Generating Map...")

    def detect_anomaly(self):
        """Function triggered when 'Detect Anomaly' is clicked."""
        self.label.config(text="Detecting Anomaly...")

    def open_manual_input(self):
        """Opens the Manual Input Screen & Closes Home Screen"""
        self.destroy()  # Close the home screen
        ManualInputScreen()

class ManualInputScreen(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set title
        self.title("Manual Data Entry")
         # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
       
        # Set dynamic window size (70% of screen resolution)
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.7)


        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Create Home Button
        home_btn = tk.Button(self, text="Home", font=("Arial", 12), command=self.go_home)
        home_btn.pack(pady=10)

        # Label
        label = tk.Label(self, text="Upload the data manually (Accepted: JPEG, MP4)", font=("Arial", 14))
        label.pack(pady=10)

        # Image Upload Button
        img_upload_btn = tk.Button(self, text="Upload Image", font=("Arial", 12), command=self.upload_image)
        img_upload_btn.pack(pady=5)

        # Video Upload Button
        video_upload_btn = tk.Button(self, text="Upload Video", font=("Arial", 12), command=self.upload_video)
        video_upload_btn.pack(pady=5)

        # Coordinates for placing dots and buttons
        #image_path = "test.jpeg"  # Change this to your image path


    def upload_image(self):
        """Open file dialog for image upload"""
        file_path = filedialog.askopenfilename(filetypes=[("JPEG Files", "*.jpeg"), ("PNG Files", "*.png")])
        if file_path:
            print(f"Image Uploaded: {file_path}")

    def upload_video(self):
        """Open file dialog for video upload"""
        file_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])
        if file_path:
            print(f"Video Uploaded: {file_path}")

    def go_home(self):
        """Return to Home Screen"""
        self.destroy()  # Close Manual Input Screen
        CrackDetector3000()  # Open Home Screen

# Run the GUI
if __name__ == '__main__':
    app = CrackDetector3000()
    app.mainloop()
