import cv2
import threading
import queue
from src.vision_ultralytics import UltralyticsVision
from src.midi_engine import MidiEngine
from src.logic import GestureLogic
from src.ui import SettingsUI
from src.scales import ScaleGenerator

class MIDIApp:
    def __init__(self):
        # Data & Logic
        self.midi_engine = MidiEngine()
        self.logic = GestureLogic(self.midi_engine)
        self.vision = UltralyticsVision()
        self.running = True
        
        # Thread-safe queue for UI image updates
        self.frame_queue = queue.Queue(maxsize=2)
        
        # UI (Settings Window)
        self.ui = SettingsUI(self.update_settings)
        
        # Processing thread
        self.process_thread = threading.Thread(target=self.vision_loop, daemon=True)

    def update_settings(self, settings):
        """Callback from UI to update engine logic."""
        print(f"Updating settings: {settings}")
        
        # Update scale
        scale_type = settings["scale_type"].lower()
        if scale_type == "pentatonic":
            scale_type = "pentatonic_minor" 
            
        self.logic.scale_type = scale_type
        
        range_preset = settings.get("range_preset", "Full Range (5 Octaves)")
        if "Melody" in range_preset:
            start_octave = 4
            num_octaves = 2
        elif "Bass" in range_preset:
            start_octave = 2
            num_octaves = 1
        else:
            start_octave = 2
            num_octaves = 5
        
        if scale_type != "theremin":
            new_scale = ScaleGenerator.get_scale(settings["root_note"], scale_type, start_octave=start_octave, num_octaves=num_octaves)
            self.logic.scale = new_scale
            print(f"Scale updated to {settings['root_note']} {scale_type} ({num_octaves} octaves)")
        else:
            print("Theremin mode active: Continuous pitch tracking.")

    def vision_loop(self):
        """Camera processing and MIDI logic."""
        print("Vision loop started.")
        import time
        try:
            while self.running:
                success, frame, result = self.vision.get_frame()
                if not success:
                    print("Vision loop: Failed to get frame.")
                    time.sleep(0.1)
                    continue
                
                # Process gestures and send MIDI
                self.logic.process(result)
                
                # Overlay zones
                h, w, _ = frame.shape
                cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 0, 0), 2)
                cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (255, 0, 0), 2)
                
                # Put frame into the queue (drop oldest if full to avoid lag)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                time.sleep(0.01) # Small delay to limit FPS/CPU
                
        except Exception as e:
            print(f"Vision loop error: {e}")
        finally:
            self.cleanup()

    def update_ui_video(self):
        """Runs in the main thread to safely update Tkinter."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.ui.update_image(frame)
        
        if self.running:
            # Re-schedule itself every 30ms (~33 FPS)
            self.ui.after(30, self.update_ui_video)

    def cleanup(self):
        self.running = False
        self.vision.release()
        self.midi_engine.close()

    def run(self):
        self.process_thread.start()
        # Start UI update polling
        self.ui.after(100, self.update_ui_video)
        # Main thread runs the settings UI
        self.ui.mainloop()
        self.cleanup()

if __name__ == "__main__":
    app = MIDIApp()
    app.run()
