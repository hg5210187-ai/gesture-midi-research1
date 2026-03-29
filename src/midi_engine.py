import mido
import time

class MidiEngine:
    def __init__(self, port_name="Indigo Gesture Controller"):
        self.port_name = port_name
        self.outport = None
        
        # Check available ports first
        available_ports = mido.get_output_names()
        print(f"Available MIDI Outputs: {available_ports}")
        
        gb_port = next((p for p in available_ports if 'GarageBand' in p), None)
        iac_port = next((p for p in available_ports if 'IAC' in p), None)
        
        try:
            if gb_port:
                self.outport = mido.open_output(gb_port)
                print(f"✅ Successfully connected to GarageBand: '{gb_port}'")
            elif iac_port:
                self.outport = mido.open_output(iac_port)
                print(f"✅ Successfully connected to macOS IAC Driver: '{iac_port}'")
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
