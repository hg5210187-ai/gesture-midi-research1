import time
import threading


class MidiLooper:
    """Records and loops back MIDI events on a dedicated channel."""

    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PLAYING = "PLAYING"

    def __init__(self, midi_engine, channel=1):
        self.midi = midi_engine
        self.channel = channel
        self.state = self.IDLE
        self.events = []
        self._record_start = 0
        self._loop_length = 0
        self._play_thread = None
        self._stop_event = threading.Event()
        self._active_notes = set()

    def toggle(self):
        """Cycle through states: IDLE -> RECORDING -> PLAYING -> IDLE"""
        if self.state == self.IDLE:
            self._start_recording()
        elif self.state == self.RECORDING:
            self._stop_recording()
        elif self.state == self.PLAYING:
            self._stop_playback()
        return self.state

    def record_event(self, event_type, **kwargs):
        """Record a MIDI event with its relative timestamp."""
        if self.state != self.RECORDING:
            return
        rel_time = time.time() - self._record_start
        self.events.append((rel_time, event_type, kwargs))

    def cleanup(self):
        """Stop everything and clean up."""
        if self.state == self.PLAYING:
            self._stop_playback()
        self.state = self.IDLE

    # --- Private ---

    def _start_recording(self):
        self.events = []
        self._record_start = time.time()
        self.state = self.RECORDING

    def _stop_recording(self):
        self._loop_length = time.time() - self._record_start
        if not self.events or self._loop_length < 0.3:
            self.state = self.IDLE
            return
        self.state = self.PLAYING
        self._start_playback()

    def _start_playback(self):
        self._stop_event.clear()
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

    def _stop_playback(self):
        self._stop_event.set()
        if self._play_thread:
            self._play_thread.join(timeout=2)
        for note in self._active_notes:
            self.midi.send_note_off(note, channel=self.channel)
        self._active_notes.clear()
        self.state = self.IDLE
        self.events = []

    def _playback_loop(self):
        while not self._stop_event.is_set():
            loop_start = time.time()

            for rel_time, event_type, data in self.events:
                if self._stop_event.is_set():
                    return
                target = loop_start + rel_time
                wait = target - time.time()
                if wait > 0:
                    if self._stop_event.wait(wait):
                        return
                self._dispatch(event_type, data)

            # Wait for remaining loop duration before repeating
            remaining = self._loop_length - (time.time() - loop_start)
            if remaining > 0:
                if self._stop_event.wait(remaining):
                    return

            # Turn off lingering notes at loop boundary
            for note in list(self._active_notes):
                self.midi.send_note_off(note, channel=self.channel)
            self._active_notes.clear()

    def _dispatch(self, event_type, data):
        if event_type == 'note_on':
            note = data['note']
            self.midi.send_note_on(note, data.get('velocity', 64), channel=self.channel)
            self._active_notes.add(note)
        elif event_type == 'note_off':
            note = data['note']
            self.midi.send_note_off(note, channel=self.channel)
            self._active_notes.discard(note)
        elif event_type == 'cc':
            self.midi.send_cc(data['control'], data['value'], channel=self.channel)
        elif event_type == 'pitch_bend':
            self.midi.send_pitch_bend(data['pitch'], channel=self.channel)
