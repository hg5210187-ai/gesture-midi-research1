import math

import cv2
import numpy as np

from src.scales import midi_to_name_octave, midi_to_freq

# Cap how many borderlines we draw for dense scales so the lines stay readable.
_MAX_LINES = 24

# Colours (BGR).
_GREEN = (0, 255, 0)        # right hand / active pitch
_ORANGE = (255, 180, 0)     # left hand
_YELLOW = (0, 255, 255)     # the pitch-sampling line

# Looper status badge colours (BGR), matching the UI's status label.
_LOOP_COLORS = {
    "IDLE": (150, 150, 150),    # gray
    "RECORDING": (0, 0, 255),   # red
    "PLAYING": (0, 200, 0),     # green
}


def _draw_bbox(frame, bbox, color, w, h):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                  color, 2)


def _draw_obb(frame, poly, color, w, h):
    """Draw a rotated (oriented) box from normalized [4,2] corner points.

    Lets the user see the right-hand OBB model's actual orientation (the angle
    that will later drive a tilt parameter), not just an axis-aligned box.
    """
    pts = (np.asarray(poly, dtype=float) * np.array([w, h])).astype(np.int32)
    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], isClosed=True,
                  color=color, thickness=2)


def _band_screen_y(y_play, margin, h):
    """Screen row for a note-space position ``y_play`` in [0, 1].

    Honours GestureLogic's ``play_margin`` so the drawn band boundaries line up
    exactly with where notes change (the dead-zone strips at top/bottom carry no
    notes). With ``margin == 0`` this reduces to the plain ``(1 - y_play) * h``.
    """
    span = 1.0 - 2.0 * margin
    y_mid = 1.0 - (margin + y_play * span)
    return y_mid * h


def _fill_band(frame, y_top, y_bot, color, alpha=0.25):
    """Blend a translucent horizontal band [y_top, y_bot) over the frame."""
    h, w = frame.shape[:2]
    y_top = max(0, min(h, int(y_top)))
    y_bot = max(0, min(h, int(y_bot)))
    if y_bot <= y_top:
        return
    roi = frame[y_top:y_bot, 0:w]
    tint = roi.copy()
    tint[:] = color
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0, roi)


def _draw_label_box(frame, text, org, fg, scale=1.0, thickness=2,
                    bg=(0, 0, 0), alpha=0.55, pad=10):
    """Draw *text* inside a translucent box whose top-left is *org*.

    The box is clamped to stay on screen. Returns ``(box_w, box_h)`` so the
    caller can position it (e.g. centre it horizontally).
    """
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    box_w = tw + 2 * pad
    box_h = th + baseline + 2 * pad

    h, w = frame.shape[:2]
    x = max(0, min(w - box_w, int(org[0])))
    y = max(0, min(h - box_h, int(org[1])))

    # Translucent background.
    roi = frame[y:y + box_h, x:x + box_w]
    bg_img = roi.copy()
    bg_img[:] = bg
    cv2.addWeighted(bg_img, alpha, roi, 1.0 - alpha, 0, roi)

    # Border + text.
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), fg, 2)
    cv2.putText(frame, text, (x + pad, y + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)
    return box_w, box_h


def _draw_status_badge(frame, text, color, org, scale=0.7, thickness=2, pad=8):
    """Solid colour-filled badge with dark text, for glanceable status.

    Drawn opaque (not translucent) so it stays legible over any background.
    """
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    box_w = tw + 2 * pad
    box_h = th + baseline + 2 * pad

    h, w = frame.shape[:2]
    x = max(0, min(w - box_w, int(org[0])))
    y = max(0, min(h - box_h, int(org[1])))

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (30, 30, 30), 1)
    cv2.putText(frame, text, (x + pad, y + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness,
                cv2.LINE_AA)
    return box_w, box_h


def _draw_rotation_gauge(frame, logic):
    """Draw a rotation dial showing the right hand's angle -> pitch mapping.

    A top semicircular gauge whose needle tracks the normalized rotation
    position (``last_angle_t`` in [0, 1]); the mode (Scale/Continuous) and the
    live angle are labelled. Used by the rotation instrument in place of the
    height-based pitch guides, which don't apply when pitch comes from rotation.
    """
    h, w = frame.shape[:2]
    t = getattr(logic, "last_angle_t", None)
    mode = getattr(logic, "rotation_mode", "scale")
    mode_txt = "CONTINUOUS" if str(mode).startswith("cont") else "SCALE"

    R = max(48, int(min(w, h) * 0.12))
    cx = w // 2
    cy = h - 24

    # Dial arc (top semicircle) + evenly spaced ticks.
    cv2.ellipse(frame, (cx, cy), (R, R), 0, 180, 360, (90, 90, 90), 2, cv2.LINE_AA)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        ang = (1.0 - tick) * math.pi
        x1 = int(cx + (R - 8) * math.cos(ang)); y1 = int(cy - (R - 8) * math.sin(ang))
        x2 = int(cx + R * math.cos(ang)); y2 = int(cy - R * math.sin(ang))
        cv2.line(frame, (x1, y1), (x2, y2), (120, 120, 120), 1, cv2.LINE_AA)

    # Needle at the current rotation position.
    if t is not None:
        ang = (1.0 - float(t)) * math.pi
        nx = int(cx + (R - 6) * math.cos(ang))
        ny = int(cy - (R - 6) * math.sin(ang))
        cv2.line(frame, (cx, cy), (nx, ny), _GREEN, 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, _GREEN, -1, cv2.LINE_AA)

    # Mode + angle label, centred just above the dial.
    deg = getattr(logic, "last_angle_deg", None)
    label = mode_txt if deg is None else f"{mode_txt}  {deg:+.0f}°"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    _draw_label_box(frame, label, (cx - (tw + 20) // 2, cy - R - 34),
                    _YELLOW, scale=0.6, thickness=2)


def render_overlay(frame, logic):
    """Draw the lightweight overlay onto *frame* in place.

    Replaces the removed YOLO skeleton render (``result.plot()``). Always draws
    the detected hand boxes for basic feedback. When "Show Note Borderlines" is
    on, draws every scale-band boundary. When "Show Current Pitch" is on, draws
    the current-pitch feedback suite: the active scale band (highlighted +
    labelled), the mid-Y line where pitch is sampled, and a large pitch readout
    box placed where the user can actually see it.
    """
    h, w = frame.shape[:2]

    # Always-on minimal feedback: hand boxes. The right hand draws its rotated
    # OBB when available (shows the model's orientation), else its axis-aligned
    # box; the left hand draws its axis-aligned pose box.
    right_obb = getattr(logic, "last_right_obb", None)
    if right_obb is not None:
        _draw_obb(frame, right_obb, _GREEN, w, h)                 # right = green (rotated)
    elif logic.last_right_bbox is not None:
        _draw_bbox(frame, logic.last_right_bbox, _GREEN, w, h)    # right = green
    if logic.last_left_bbox is not None:
        _draw_bbox(frame, logic.last_left_bbox, _ORANGE, w, h)    # left = orange

    # Always-on looper status badge (top-left), colour-coded like the UI label,
    # so the state is visible on the video feed while the left hand drives it.
    looper = getattr(logic, "looper", None)
    if looper is not None and getattr(looper, "state", None):
        state = looper.state
        _draw_status_badge(frame, f"LOOP: {state}",
                           _LOOP_COLORS.get(state, (150, 150, 150)), (10, 12))

    is_theremin = (logic.scale_type == "theremin")

    # The rotation instrument reads pitch from the hand's ANGLE, not its height,
    # so the height-based guides below are skipped in favour of a rotation dial.
    pitch_from_rotation = getattr(logic, "pitch_from_rotation", False)

    # Playable dead-zone margin used by GestureLogic to make the extreme notes
    # reachable; the borderlines/region below honour it so they stay aligned.
    margin = getattr(logic, "play_margin", 0.0)

    # --- Note borderlines ---
    # Band i (between boundaries i and i+1) plays scale[i], matching
    # GestureLogic.map_pitch's note_index = int(y * len(scale)). The boundary
    # at note-space y = i/n maps to a screen row via _band_screen_y.
    if (logic.show_borderlines and not pitch_from_rotation
            and not is_theremin and logic.scale):
        n = len(logic.scale)
        step = max(1, (n + _MAX_LINES - 1) // _MAX_LINES)   # thin dense scales
        label = (n <= 24)
        for i in range(1, n):
            if i % step != 0:
                continue
            screen_y = int(_band_screen_y(i / n, margin, h))
            octave_line = (logic.scale_type == "chromatic" and i % 12 == 0)
            color = (0, 200, 255) if octave_line else (90, 90, 90)
            cv2.line(frame, (0, screen_y), (w, screen_y), color,
                     2 if octave_line else 1)
            if label:
                cv2.putText(frame, midi_to_name_octave(logic.scale[i]),
                            (5, max(12, screen_y - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # --- Current-pitch feedback suite ---
    # Pitch is read from the vertical MIDPOINT of the right-hand box, so three
    # guides make that legible: (1) the active scale band, highlighted and
    # labelled so the user sees where to place their hand; (2) the mid-Y line
    # marking exactly where pitch is sampled; (3) a large pitch readout box.
    if logic.show_pitch:
        # (1) Highlight + label the band the hand is currently in (scale mode).
        if (not pitch_from_rotation and not is_theremin and logic.scale
                and logic.last_hand_y_norm is not None):
            n = len(logic.scale)
            idx = max(0, min(n - 1, int(logic.last_hand_y_norm * n)))
            band_top = _band_screen_y((idx + 1) / n, margin, h)
            band_bot = _band_screen_y(idx / n, margin, h)
            _fill_band(frame, band_top, band_bot, _GREEN, alpha=0.25)
            band_name = midi_to_name_octave(logic.scale[idx])
            label_y = int((band_top + band_bot) / 2.0)
            cv2.putText(frame, f"> {band_name}", (10, max(22, label_y + 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, _GREEN, 2, cv2.LINE_AA)

        # (2) The mid-Y line where pitch is actually determined (height versions).
        if not pitch_from_rotation and logic.last_right_bbox is not None:
            x1, y1, x2, y2 = logic.last_right_bbox
            y_mid = int((y1 + y2) / 2.0 * h)
            cv2.line(frame, (0, y_mid), (w, y_mid), _YELLOW, 2, cv2.LINE_AA)
            cv2.putText(frame, "pitch", (w - 75, max(16, y_mid - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _YELLOW, 1, cv2.LINE_AA)

        # (2b) Rotation instrument: a rotation dial in place of the height guides.
        if pitch_from_rotation:
            _draw_rotation_gauge(frame, logic)

        # (3) Large pitch readout box, centred near the top.
        if logic.current_pitch_note != -1:
            note = logic.current_pitch_note
            txt = f"{midi_to_name_octave(note)}  ({midi_to_freq(note):.0f} Hz)"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            box_w = tw + 20
            _draw_label_box(frame, txt, ((w - box_w) // 2, 12),
                            _GREEN, scale=1.0, thickness=2)

    return frame
