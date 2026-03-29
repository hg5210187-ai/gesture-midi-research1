import mido

class ScaleGenerator:
    """Generates MIDI note numbers for various scales across all keys."""
    
    # Intervals in semitones
    INTERVALS = {
        "chromatic": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "pentatonic_major": [2, 2, 3, 2, 3],
        "pentatonic_minor": [3, 2, 2, 3, 2]
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
        Example: get_scale("A", "pentatonic_minor")
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

if __name__ == "__main__":
    # Test
    a_minor_pent = ScaleGenerator.get_scale("A", "pentatonic_minor")
    print(f"A Minor Pentatonic: {a_minor_pent}")
    
    c_chromatic = ScaleGenerator.get_scale("C", "chromatic", num_octaves=1)
    print(f"C Chromatic (1 Octave): {c_chromatic}")
