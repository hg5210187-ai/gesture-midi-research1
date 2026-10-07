import queue
import threading

import customtkinter as ctk
from PIL import Image


# General MIDI instruments (display name -> GM program number 0-127). A curated
# spread of common sounds so switching is one click. Sent as a Program Change;
# see MidiEngine.send_program_change for the GarageBand caveat.
INSTRUMENTS = {
    "Grand Piano": 0,
    "Electric Piano": 4,
    "Harpsichord": 6,
    "Vibraphone": 11,
    "Drawbar Organ": 16,
    "Church Organ": 19,
    "Accordion": 21,
    "Nylon Guitar": 24,
    "Steel Guitar": 25,
    "Jazz Guitar": 26,
    "Clean Electric Guitar": 27,
    "Overdrive Guitar": 29,
    "Distortion Guitar": 30,
    "Acoustic Bass": 32,
    "Fingered Bass": 33,
    "Slap Bass": 36,
    "Violin": 40,
    "Cello": 42,
    "String Ensemble": 48,
    "Choir Aahs": 52,
    "Trumpet": 56,
    "Trombone": 57,
    "French Horn": 60,
    "Alto Sax": 65,
    "Tenor Sax": 66,
    "Oboe": 68,
    "Clarinet": 71,
    "Flute": 73,
    "Pan Flute": 75,
    "Synth Lead (Square)": 80,
    "Synth Lead (Saw)": 81,
    "Synth Pad (Warm)": 89,
    "Synth Pad (Halo)": 94,
}


class SettingsUI(ctk.CTk):
    def __init__(self, on_change_callback, on_calibrate=None,
                 show_rotation_mode=False, on_calibrate_rotation=None):
        super().__init__()
        # ``show_rotation_mode`` reveals the Scale/Continuous toggle used only by
        # the rotation instrument (main_rotate.py). Off for the other versions so
        # their sidebar is unchanged. ``on_calibrate_rotation`` (called with
        # "low"/"high"/"reset") adds the rotation-range calibration buttons.
        self._show_rotation_mode = show_rotation_mode
        self.on_calibrate_rotation = on_calibrate_rotation
        self.title("MIDI Gesture Settings")
        self.geometry("1280x800")
        self.on_change_callback = on_change_callback
        self.on_calibrate = on_calibrate

        # Settings changes (dropdowns/sliders) are applied on a dedicated worker
        # thread so the Tk main thread never blocks on the callback work (MIDI
        # program change, scale regeneration) — that blocking is what made the
        # Instrument dropdown feel laggy. Widget values are still read on the
        # main thread in _update_settings (Tk is not thread-safe); only the
        # non-UI on_change_callback runs off-thread. Rapid changes coalesce to
        # the most recent queued settings so a fast slider drag can't back up.
        self._settings_queue = queue.Queue()
        self._settings_thread = threading.Thread(target=self._settings_worker, daemon=True)
        self._settings_thread.start()

        self._fullscreen = False
        self._sidebar_visible = True

        # Open maximised so the camera feed is large by default. True OS
        # fullscreen is available via the Fullscreen button / F11 (Esc exits).
        self.after(120, self._maximize)
        self.bind("<F11>", lambda _e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda _e: self._set_fullscreen(False))

        # Layout: row 0 = top bar (full width), row 1 = sidebar + video.
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)   # sidebar keeps its width
        self.grid_columnconfigure(1, weight=1)   # video takes the rest

        # ---------------------------------------------------------------- #
        #  Top bar: sidebar toggle, "always show" pin, fullscreen           #
        # ---------------------------------------------------------------- #
        self.topbar = ctk.CTkFrame(self, height=52)
        self.topbar.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="ew")

        self.toggle_btn = ctk.CTkButton(self.topbar, text="☰  Hide Settings",
                                        width=160, command=self.toggle_sidebar)
        self.toggle_btn.pack(side="left", padx=10, pady=8)

        self.pin_check = ctk.CTkCheckBox(self.topbar, text="Always show settings",
                                         command=self._on_pin_toggle)
        self.pin_check.select()   # default: sidebar pinned open (legacy behaviour)
        self.pin_check.pack(side="left", padx=10)

        self.fullscreen_btn = ctk.CTkButton(self.topbar, text="⛶  Fullscreen",
                                            width=150, command=self.toggle_fullscreen)
        self.fullscreen_btn.pack(side="right", padx=10, pady=8)

        # With the sidebar pinned by default, the toggle button is inactive.
        self.toggle_btn.configure(state="disabled")

        # ---------------------------------------------------------------- #
        #  Settings Panel (Left, scrollable)                                #
        # ---------------------------------------------------------------- #
        self.settings_frame = ctk.CTkScrollableFrame(self, width=300)
        self.settings_frame.grid(row=1, column=0, padx=(10, 5), pady=10, sticky="nsew")

        # Rotation Mode (rotation instrument only): Scale snaps rotation through
        # the scale; Continuous glides the pitch smoothly.
        if self._show_rotation_mode:
            self.rotation_mode_label = ctk.CTkLabel(
                self.settings_frame, text="Rotation Mode", font=("Arial", 16, "bold"))
            self.rotation_mode_label.pack(padx=20, pady=(15, 5), anchor="w")

            self.rotation_mode = ctk.CTkSegmentedButton(
                self.settings_frame, values=["Scale", "Continuous"],
                command=self._update_settings)
            self.rotation_mode.set("Scale")
            self.rotation_mode.pack(padx=20, pady=(0, 5), fill="x")

            self.rotation_mode_hint = ctk.CTkLabel(
                self.settings_frame,
                text="Rotate the right hand to play the pitch",
                font=("Arial", 11), text_color="gray60")
            self.rotation_mode_hint.pack(padx=20, pady=(0, 10))

            # Rotation-range calibration: capture the angle for the lowest note,
            # then the highest, so the pitch range fits your natural rotation.
            if self.on_calibrate_rotation is not None:
                self.rot_calib_label = ctk.CTkLabel(
                    self.settings_frame, text="Rotation Range",
                    font=("Arial", 16, "bold"))
                self.rot_calib_label.pack(padx=20, pady=(10, 5), anchor="w")

                rot_btn_row = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
                rot_btn_row.pack(padx=20, pady=(0, 5), fill="x")
                self.rot_low_btn = ctk.CTkButton(
                    rot_btn_row, text="Set Low Note", width=120,
                    command=lambda: self._calibrate_rotation("low"))
                self.rot_low_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
                self.rot_high_btn = ctk.CTkButton(
                    rot_btn_row, text="Set High Note", width=120,
                    command=lambda: self._calibrate_rotation("high"))
                self.rot_high_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))

                self.rot_reset_btn = ctk.CTkButton(
                    self.settings_frame, text="Reset Range", fg_color="gray30",
                    hover_color="gray25",
                    command=lambda: self._calibrate_rotation("reset"))
                self.rot_reset_btn.pack(padx=20, pady=(4, 4), fill="x")

                self.rot_calib_status = ctk.CTkLabel(
                    self.settings_frame,
                    text="Hold hand at lowest-pitch tilt → Set Low, "
                         "then rotate → Set High",
                    font=("Arial", 11), text_color="gray60", wraplength=260)
                self.rot_calib_status.pack(padx=20, pady=(0, 10))

        self.scale_label = ctk.CTkLabel(self.settings_frame, text="Scale Type", font=("Arial", 16, "bold"))
        self.scale_label.pack(padx=20, pady=(15, 5), anchor="w")

        scale_options = [
            "Chromatic", "Major", "Minor",
            "Ionian", "Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian",
            "Major Pentatonic", "Minor Pentatonic", "Theremin"
        ]
        self.scale_type = ctk.CTkOptionMenu(self.settings_frame, values=scale_options,
                                            command=self._update_settings)
        self.scale_type.set("Chromatic")
        self.scale_type.pack(padx=20, pady=10, fill="x")

        self.root_label = ctk.CTkLabel(self.settings_frame, text="Root Note", font=("Arial", 16, "bold"))
        self.root_label.pack(padx=20, pady=(20, 5), anchor="w")

        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.root_note = ctk.CTkOptionMenu(self.settings_frame, values=notes, command=self._update_settings)
        self.root_note.set("A")
        self.root_note.pack(padx=20, pady=10, fill="x")

        # Instrument (MIDI Program Change)
        self.instrument_label = ctk.CTkLabel(self.settings_frame, text="Instrument", font=("Arial", 16, "bold"))
        self.instrument_label.pack(padx=20, pady=(20, 5), anchor="w")

        self.instrument = ctk.CTkOptionMenu(self.settings_frame, values=list(INSTRUMENTS.keys()),
                                            command=self._update_settings)
        self.instrument.set("Grand Piano")
        self.instrument.pack(padx=20, pady=10, fill="x")

        # Pitch Range
        self.range_label = ctk.CTkLabel(self.settings_frame, text="Pitch Range", font=("Arial", 16, "bold"))
        self.range_label.pack(padx=20, pady=(20, 5), anchor="w")

        ranges = ["Full Range (5 Octaves)", "Melody (2 Octaves)", "Melody (1 Octave)", "Bass (1 Octave)"]
        self.range_preset = ctk.CTkOptionMenu(self.settings_frame, values=ranges, command=self._update_settings)
        self.range_preset.set("Full Range (5 Octaves)")
        self.range_preset.pack(padx=20, pady=10, fill="x")

        self.track_label = ctk.CTkLabel(self.settings_frame, text="Number of Tracks (1-5)", font=("Arial", 16, "bold"))
        self.track_label.pack(padx=20, pady=(20, 5), anchor="w")

        self.track_slider = ctk.CTkSlider(self.settings_frame, from_=1, to=5, number_of_steps=4, command=self._update_settings)
        self.track_slider.set(3)
        self.track_slider.pack(padx=20, pady=10, fill="x")

        # Volume calibration
        self.calibrate_btn = ctk.CTkButton(self.settings_frame, text="Calibrate Min",
                                           command=self._calibrate)
        self.calibrate_btn.pack(padx=20, pady=(20, 5), fill="x")

        self.calib_status = ctk.CTkLabel(self.settings_frame,
                                         text="Hold hand at min size, then calibrate",
                                         font=("Arial", 11), text_color="gray60")
        self.calib_status.pack(padx=20, pady=(0, 5))

        self.vol_label = ctk.CTkLabel(self.settings_frame, text="Volume Sensitivity",
                                      font=("Arial", 16, "bold"))
        self.vol_label.pack(padx=20, pady=(15, 5), anchor="w")

        self.volume_sensitivity = ctk.CTkSlider(self.settings_frame, from_=0.5, to=8.0,
                                                command=self._update_settings)
        self.volume_sensitivity.set(2.0)
        self.volume_sensitivity.pack(padx=20, pady=10, fill="x")

        # Volume gate: below this MIDI volume the note is fully cut, so fast
        # in/out hand movements switch the sound cleanly on and off.
        self.vol_gate_label = ctk.CTkLabel(self.settings_frame, text="Volume Gate (cut-off): 8",
                                           font=("Arial", 16, "bold"))
        self.vol_gate_label.pack(padx=20, pady=(15, 5), anchor="w")

        self.volume_gate = ctk.CTkSlider(self.settings_frame, from_=0, to=40, number_of_steps=40,
                                         command=self._update_settings)
        self.volume_gate.set(8)
        self.volume_gate.pack(padx=20, pady=(0, 5), fill="x")

        self.vol_gate_hint = ctk.CTkLabel(self.settings_frame,
                                          text="Higher = cuts off sooner (snappier on/off)",
                                          font=("Arial", 11), text_color="gray60")
        self.vol_gate_hint.pack(padx=20, pady=(0, 10))

        # Overlay toggles
        self.show_borderlines_check = ctk.CTkCheckBox(self.settings_frame,
                text="Show Note Borderlines", command=self._update_settings)
        self.show_borderlines_check.pack(padx=20, pady=(10, 5), anchor="w")

        self.show_pitch_check = ctk.CTkCheckBox(self.settings_frame,
                text="Show Current Pitch", command=self._update_settings)
        self.show_pitch_check.select()   # on by default (matches GestureLogic)
        self.show_pitch_check.pack(padx=20, pady=(5, 10), anchor="w")

        # Looper status
        self.looper_label = ctk.CTkLabel(self.settings_frame, text="Looper: Idle",
                                          font=("Arial", 14, "bold"), text_color="gray")
        self.looper_label.pack(padx=20, pady=(20, 5))

        self.looper_hint = ctk.CTkLabel(self.settings_frame,
                                         text="Left hand fist to toggle",
                                         font=("Arial", 11), text_color="gray60")
        self.looper_hint.pack(padx=20, pady=(0, 10))

        self.status_text = ctk.CTkLabel(self.settings_frame, text="Status: Running", text_color="green")
        self.status_text.pack(padx=20, pady=(10, 30))

        # ---------------------------------------------------------------- #
        #  Video Panel (Right)                                              #
        # ---------------------------------------------------------------- #
        self.video_frame = ctk.CTkFrame(self, fg_color="black")
        self.video_frame.grid(row=1, column=1, padx=(5, 10), pady=10, sticky="nsew")
        self.video_frame.pack_propagate(False)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Camera Feed Loading...", text_color="white")
        self.video_label.pack(expand=True, fill="both")

    # ------------------------------------------------------------------ #
    #  Sidebar show/hide + pin                                            #
    # ------------------------------------------------------------------ #
    def toggle_sidebar(self):
        """Show or hide the settings sidebar (the video expands to fill)."""
        if self._sidebar_visible:
            self.settings_frame.grid_remove()
            self._sidebar_visible = False
            self.toggle_btn.configure(text="☰  Show Settings")
        else:
            self.settings_frame.grid()
            self._sidebar_visible = True
            self.toggle_btn.configure(text="☰  Hide Settings")

    def _on_pin_toggle(self):
        """'Always show settings' checkbox: pin the sidebar open or free it."""
        if self.pin_check.get():
            if not self._sidebar_visible:
                self.toggle_sidebar()
            self.toggle_btn.configure(state="disabled")
        else:
            self.toggle_btn.configure(state="normal")

    # ------------------------------------------------------------------ #
    #  Fullscreen / window sizing                                         #
    # ------------------------------------------------------------------ #
    def _maximize(self):
        """Fill (most of) the screen, leaving room for the macOS menu bar."""
        try:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            self.geometry(f"{sw}x{max(400, sh - 80)}+0+0")
        except Exception as e:
            print(f"Maximize failed: {e}")

    def toggle_fullscreen(self):
        self._set_fullscreen(not self._fullscreen)

    def _set_fullscreen(self, on):
        self._fullscreen = bool(on)
        try:
            self.attributes("-fullscreen", self._fullscreen)
        except Exception as e:
            print(f"Fullscreen toggle failed: {e}")
        self.fullscreen_btn.configure(
            text="⛶  Exit Fullscreen" if self._fullscreen else "⛶  Fullscreen")

    # ------------------------------------------------------------------ #
    #  Video                                                              #
    # ------------------------------------------------------------------ #
    def update_image(self, cv_frame):
        """Convert CV frame to CTkImage and display it, scaled to the panel."""
        try:
            import cv2
            h, w = cv_frame.shape[:2]
            rgb_image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Fit within the current video panel, preserving aspect ratio, so
            # the feed grows with the window / fullscreen instead of staying
            # fixed at a small size.
            avail_w = self.video_frame.winfo_width() - 16
            avail_h = self.video_frame.winfo_height() - 16
            if avail_w <= 1 or avail_h <= 1:
                # Before the first layout pass winfo_* is 1; fall back to a
                # sensible default until real sizes are known.
                avail_w, avail_h = 700, int(h * (700 / w))

            scale = min(avail_w / w, avail_h / h)
            target_w = max(1, int(w * scale))
            target_h = max(1, int(h * scale))

            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(target_w, target_h))

            self.video_label.configure(image=ctk_image, text="")
            self.video_label._image = ctk_image
        except Exception as e:
            print(f"UI Update Error: {e}")

    def update_looper_status(self, state):
        """Update the looper status label."""
        colors = {"IDLE": "gray", "RECORDING": "red", "PLAYING": "green"}
        self.looper_label.configure(
            text=f"Looper: {state.capitalize()}",
            text_color=colors.get(state, "gray")
        )

    # ------------------------------------------------------------------ #
    #  Settings plumbing                                                  #
    # ------------------------------------------------------------------ #
    def _settings_worker(self):
        """Apply queued settings off the Tk main thread, newest-wins.

        Only ``on_change_callback`` (which touches engine/logic + MIDI, never Tk)
        runs here. A ``None`` item is the shutdown sentinel.
        """
        while True:
            settings = self._settings_queue.get()
            if settings is None:
                break
            # Coalesce: skip to the most recent queued settings so a fast slider
            # drag doesn't process every intermediate value and fall behind.
            while not self._settings_queue.empty():
                try:
                    nxt = self._settings_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    return
                settings = nxt
            try:
                self.on_change_callback(settings)
            except Exception as e:
                print(f"Settings update error: {e}")

    def _update_settings(self, _=None):
        # Read widget values on the main thread (Tk is not thread-safe), then
        # hand the plain dict to the worker thread so the UI doesn't block.
        gate = int(self.volume_gate.get())
        self.vol_gate_label.configure(text=f"Volume Gate (cut-off): {gate}")

        settings = {
            "scale_type": self.scale_type.get(),
            "root_note": self.root_note.get(),
            "range_preset": self.range_preset.get(),
            "instrument_program": INSTRUMENTS[self.instrument.get()],
            "tracks": int(self.track_slider.get()),
            "volume_gain": float(self.volume_sensitivity.get()),
            "volume_gate": gate,
            "show_borderlines": bool(self.show_borderlines_check.get()),
            "show_pitch": bool(self.show_pitch_check.get()),
        }
        # Rotation instrument only: report the Scale/Continuous mode.
        if self._show_rotation_mode:
            settings["rotation_mode"] = self.rotation_mode.get()
        self._settings_queue.put(settings)

    def _calibrate(self):
        if self.on_calibrate:
            self.on_calibrate()

    def set_calib_status(self, text, ok=True):
        """Update the calibration feedback label (called from the main thread)."""
        self.calib_status.configure(text=text,
                                    text_color="green" if ok else "orange")

    def _calibrate_rotation(self, which):
        if self.on_calibrate_rotation:
            self.on_calibrate_rotation(which)

    def set_rotation_calib_status(self, text, ok=True):
        """Update the rotation-range calibration feedback label (main thread)."""
        if hasattr(self, "rot_calib_status"):
            self.rot_calib_status.configure(
                text=text, text_color="green" if ok else "orange")
