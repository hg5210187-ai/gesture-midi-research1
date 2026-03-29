import customtkinter as ctk
from PIL import Image

class SettingsUI(ctk.CTk):
    def __init__(self, on_change_callback):
        super().__init__()
        self.title("MIDI Gesture Settings")
        self.geometry("1100x700")
        self.on_change_callback = on_change_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3) 
        self.grid_rowconfigure(0, weight=1)
        
        # --- Settings Panel (Left) ---
        self.settings_frame = ctk.CTkFrame(self, width=300)
        self.settings_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.scale_label = ctk.CTkLabel(self.settings_frame, text="Scale Type", font=("Arial", 16, "bold"))
        self.scale_label.pack(padx=20, pady=(20, 5), anchor="w")
        
        self.scale_type = ctk.CTkSegmentedButton(self.settings_frame, values=["Chromatic", "Pentatonic", "Theremin"],
                                               command=self._update_settings)
        self.scale_type.set("Chromatic")
        self.scale_type.pack(padx=20, pady=10, fill="x")

        self.root_label = ctk.CTkLabel(self.settings_frame, text="Root Note", font=("Arial", 16, "bold"))
        self.root_label.pack(padx=20, pady=(20, 5), anchor="w")
        
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.root_note = ctk.CTkOptionMenu(self.settings_frame, values=notes, command=self._update_settings)
        self.root_note.set("A")
        self.root_note.pack(padx=20, pady=10, fill="x")

        # Pitch Range
        self.range_label = ctk.CTkLabel(self.settings_frame, text="Pitch Range", font=("Arial", 16, "bold"))
        self.range_label.pack(padx=20, pady=(20, 5), anchor="w")
        
        ranges = ["Full Range (5 Octaves)", "Melody (2 Octaves)", "Bass (1 Octave)"]
        self.range_preset = ctk.CTkOptionMenu(self.settings_frame, values=ranges, command=self._update_settings)
        self.range_preset.set("Full Range (5 Octaves)")
        self.range_preset.pack(padx=20, pady=10, fill="x")

        self.track_label = ctk.CTkLabel(self.settings_frame, text="Number of Tracks (1-5)", font=("Arial", 16, "bold"))
        self.track_label.pack(padx=20, pady=(20, 5), anchor="w")
        
        self.track_slider = ctk.CTkSlider(self.settings_frame, from_=1, to=5, number_of_steps=4, command=self._update_settings)
        self.track_slider.set(3)
        self.track_slider.pack(padx=20, pady=10, fill="x")

        self.status_text = ctk.CTkLabel(self.settings_frame, text="Status: Running", text_color="green")
        self.status_text.pack(padx=20, pady=30)

        # --- Video Panel (Right) ---
        self.video_frame = ctk.CTkFrame(self, fg_color="black")
        self.video_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.video_frame.pack_propagate(False)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Camera Feed Loading...", text_color="white")
        self.video_label.pack(expand=True, fill="both")

    def update_image(self, cv_frame):
        """Convert CV frame to CTkImage and display it."""
        try:
            import cv2
            h, w = cv_frame.shape[:2]
            rgb_image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            target_w = 700
            target_h = int(h * (target_w / w))
            
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(target_w, target_h))
            
            self.video_label.configure(image=ctk_image, text="")
            self.video_label._image = ctk_image
        except Exception as e:
            print(f"UI Update Error: {e}")

    def _update_settings(self, _=None):
        settings = {
            "scale_type": self.scale_type.get(),
            "root_note": self.root_note.get(),
            "range_preset": self.range_preset.get(),
            "tracks": int(self.track_slider.get())
        }
        self.on_change_callback(settings)
