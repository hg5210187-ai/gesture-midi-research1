import mido
import os
import time

class MidiEngine:
    def __init__(self, port_name="Indigo Gesture Controller"):
        self.port_name = port_name
        self.outport = None
        
        # Check available ports first
        available_ports = mido.get_output_names()
        print(f"Available MIDI Outputs: {available_ports}")
        
        # MIDI_PORT picks a specific output by (partial) name. Otherwise use the
        # macOS IAC Driver bus, which any DAW (Cubase, GarageBand, ...) can
        # listen to as a MIDI input.
        wanted = os.environ.get("MIDI_PORT")
        wanted_port = next((p for p in available_ports if wanted and wanted in p), None)
        if wanted and not wanted_port:
            print(f"⚠️ MIDI_PORT '{wanted}' not found; using the default port.")
        iac_port = next((p for p in available_ports if 'IAC' in p), None)
        
        try:
            if wanted_port or iac_port:
                port = wanted_port or iac_port
                self.outport = mido.open_output(port)
                print(f"✅ Connected to MIDI output: '{port}'")
            else:
                # Create a virtual port if supported by backend (e.g. CoreMIDI on macOS)
                self.outport = mido.open_output(self.port_name, virtual=True)
                print(f"✅ Virtual MIDI port '{self.port_name}' created successfully.")
        except Exception as e:
            print(f"❌ Failed to connect or create virtual MIDI port: {e}")
            if available_ports:
                self.outport = mido.open_output(available_ports[0])
                print(f"⚠️ Opened fallback MIDI port: {available_ports[0]}")
            else:
                print("❌ No MIDI ports available.")

    def send_note_on(self, note, velocity=64, channel=0):
        if self.outport:
            msg = mido.Message('note_on', note=note, velocity=velocity, channel=channel)
            self.outport.send(msg)

    def send_note_off(self, note, velocity=0, channel=0):
        if self.outport:
            msg = mido.Message('note_off', note=note, velocity=velocity, channel=channel)
            self.outport.send(msg)
            
    def send_cc(self, control, value, channel=0):
        """Send a Control Change message (e.g., volume CC=7)."""
        if self.outport:
            msg = mido.Message('control_change', control=control, value=value, channel=channel)
            self.outport.send(msg)

    def send_program_change(self, program, channel=0):
        """Switch the instrument/patch on a channel (General MIDI program 0-127).

        Note: this works with destinations that honour Program Change (e.g.
        HALion Sonic in Cubase, or any General MIDI synth). GarageBand has a
        non-standard MIDI implementation and ignores it.
        """
        if self.outport:
            program = max(0, min(127, int(program)))
            msg = mido.Message('program_change', program=program, channel=channel)
            self.outport.send(msg)

    def send_pitch_bend(self, pitch, channel=0):
        """Pitch is between -8192 and 8191."""
        if self.outport:
            # Clamp pitch just to be safe
            pitch = max(-8192, min(8191, pitch))
            msg = mido.Message('pitchwheel', pitch=pitch, channel=channel)
            self.outport.send(msg)

    def close(self):
        if self.outport:
            self.outport.close()

if __name__ == "__main__":
    # Test the MIDI Engine
    engine = MidiEngine()
    print("Sending Note On 60")
    engine.send_note_on(60)
    time.sleep(1)
    print("Sending Note Off 60")
    engine.send_note_off(60)
    engine.close()
