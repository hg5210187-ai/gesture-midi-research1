import mido

class ScaleGenerator:
    """Generates MIDI note numbers for various scales across all keys."""
    
    # Intervals in semitones
    INTERVALS = {
        "chromatic": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "major": [2, 2, 1, 2, 2, 2, 1],
        "minor": [2, 1, 2, 2, 1, 2, 2],
        "ionian": [2, 2, 1, 2, 2, 2, 1],
        "dorian": [2, 1, 2, 2, 2, 1, 2],
        "phrygian": [1, 2, 2, 2, 1, 2, 2],
        "lydian": [2, 2, 2, 1, 2, 2, 1],
        "mixolydian": [2, 2, 1, 2, 2, 1, 2],
        "aeolian": [2, 1, 2, 2, 1, 2, 2],
        "locrian": [1, 2, 2, 1, 2, 2, 2],
        "major_pentatonic": [2, 2, 3, 2, 3],
        "minor_pentatonic": [3, 2, 2, 3, 2]
    }
    
    # Note name to semitone offset from C
    NOTE_MAP = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, 
        "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, 
        "A#": 10, "Bb": 10, "B": 11
    }

    @staticmethod
    def get_scale(root_note_name, scale_type, start_octave=2, num_octaves=5):
        """
        Returns a list of MIDI note numbers for the specified scale.
        Example: get_scale("A", "minor_pentatonic")
        """
        root_offset = ScaleGenerator.NOTE_MAP.get(root_note_name, 0)
        base_note = (start_octave + 1) * 12 + root_offset
        
        intervals = ScaleGenerator.INTERVALS.get(scale_type, ScaleGenerator.INTERVALS["chromatic"])
        
        scale_notes = []
        current_note = base_note
        
        for _ in range(num_octaves):
            for interval in intervals:
                if current_note <= 127:
                    scale_notes.append(current_note)
                current_note += interval
                
        return scale_notes


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name_octave(note):
    """Convert a MIDI note number to a name+octave string (69 -> 'A4')."""
    note = int(note)
    return f"{_NOTE_NAMES[note % 12]}{(note // 12) - 1}"


def midi_to_freq(note):
    """Convert a MIDI note number to its equal-tempered frequency in Hz (69 -> 440.0)."""
    return 440.0 * (2.0 ** ((int(note) - 69) / 12.0))


if __name__ == "__main__":
    # Test
    a_minor_pent = ScaleGenerator.get_scale("A", "minor_pentatonic")
    print(f"A Minor Pentatonic: {a_minor_pent}")
    
    c_chromatic = ScaleGenerator.get_scale("C", "chromatic", num_octaves=1)
    print(f"C Chromatic (1 Octave): {c_chromatic}")
