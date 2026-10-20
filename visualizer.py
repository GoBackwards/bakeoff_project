"""Clink: ask for help from bed by tapping what's on the nightstand.

Desktop app for our vibration bakeoff project (built on the course starter). The laptop microphone
hears which object rings after a spoon tap: the steel bottle (Interaction#1, "Water, please") or the
ceramic bowl (Interaction#2, "Please come"). Everything else is Neutral.

Three screens, switched with the buttons at the top:
  SETUP  microphone, Calibrate Quiet, recognition settings, bakeoff roles, a demo checklist
  TRAIN  record each class, train / save / load the model
  DEMO   the product view and the bakeoff trial recorder: SPACE starts a trial, SPACE again stops it
         and shows the set of labels seen in between and whether the trial passed

Run with:  uv run visualizer.py
"""
import json
import os
import pickle
import queue
import re
import tempfile
import threading
import time
from math import cos, pi, sin
from pathlib import Path

import numpy as np
import pyaudio

import dearpygui.dearpygui as dpg

import trial as T
from models import (
    train_model,
    load_model,
    save_model,
    get_ear,
    get_audio_features,
    preprocess_data,
    predict_confident,
    frame_level_db,
    calibrate_noise,
    neutral_class,
    LabelSmoother,
    GATE_MARGIN_DB,
    DEFAULT_MODEL_TYPE,
    MODEL_OPTIONS,
)

# ——— Screen text: rename things here (ASCII only: the font atlas holds characters 0x20-0xFF) ———

PRODUCT = {
    "name": "Clink",
    "tagline": "Ask for help from bed by tapping what's on the nightstand.",
    "hint": "One firm spoon tap, then let it ring. Gentle taps are ignored.",
    T.NEUTRAL: {"name": "Neutral", "request": "Listening"},
    T.I1: {"name": "Bottle", "request": "Water, please", "how": "TAP THE BOTTLE", "counter": "water requests",
           "ring_hz": (360, 740, 1360)},
    T.I2: {"name": "Bowl", "request": "Please come", "how": "TAP THE BOWL", "counter": "calls",
           "ring_hz": (640, 1280, 3100)},
}

# What the experimenter reads to the participant before each trial type.
SAY = {
    T.PRACTICE_NEUTRAL: 'Practice: "For a few seconds, do anything except tap the bottle or the bowl."  (you may help)',
    T.PRACTICE_I1: 'Practice: "Hold the spoon like a little hammer. One firm tap on the side of the bottle, at the '
                   'dot, and let it ring."  (you may help)',
    T.PRACTICE_I2: 'Practice: "Now the bowl: one firm tap on the side, at the dot, and let it ring."  (you may help)',
    T.POS_I1: 'Say only: "Tap the bottle."  (no help)',
    T.POS_I2: 'Say only: "Tap the bowl."  (no help)',
    T.NEG_I1: 'Say only: "Anything but the bottle."  (tapping the bowl is allowed; no help)',
    T.NEG_I2: 'Say only: "Anything but the bowl."  (tapping the bottle is allowed; no help)',
}
TYPE_KEY = {T.PRACTICE_NEUTRAL: "PN", T.PRACTICE_I1: "P1", T.PRACTICE_I2: "P2",
            T.POS_I1: "POS1", T.POS_I2: "POS2", T.NEG_I1: "NEG1", T.NEG_I2: "NEG2"}
TALLY_CELL = {T.POS_I1: (0, 1), T.POS_I2: (0, 2), T.NEG_I1: (1, 1), T.NEG_I2: (1, 2)}  # (row, column)
KEY_HINTS = "DEMO keys:  SPACE start/stop trial   Shift+SPACE start anyway   Esc cancel trial   F11 full screen"

# ——— Look ———

C = {
    "bg": (15, 17, 21), "card": (23, 26, 33), "card2": (30, 34, 43), "frame": (37, 41, 54),
    "frame_hi": (48, 53, 69), "frame_act": (58, 64, 83), "border": (48, 53, 66),
    "text": (231, 234, 240), "muted": (139, 146, 160),
    "accent": (129, 140, 248), "accent_fill": (79, 70, 229),
    "neutral": (156, 163, 175), "i1": (56, 189, 248), "i2": (251, 146, 60),  # bottle blue, bowl amber
    "ok": (21, 128, 61), "bad": (185, 28, 28), "rec": (225, 29, 72), "ready": (51, 65, 85), "wait": (161, 98, 7),
    "info": (30, 64, 175), "warn_text": (250, 204, 21), "ok_text": (74, 222, 128), "bad_text": (248, 113, 113),
}
ROLE_TAG = {T.NEUTRAL: "n", T.I1: "i1", T.I2: "i2"}
ROLE_COLOR = {T.NEUTRAL: C["neutral"], T.I1: C["i1"], T.I2: C["i2"]}
MODES = ("SETUP", "TRAIN", "DEMO")
TOPBAR_H, HERO_H, FOOTER_H = 50, 130, 60
METER_W, METER_H, METER_TOP_DB, PEAK_FALL_DB_S = 380, 22, 40.0, 20.0
TILE_H, LOG_ROWS = 220, 2
PLOT_MAX_HZ = 10_000  # the objects ring well below this; it is also the top of the gate band
REQUEST_SHOW_S = 3.0  # a request stays on the product view this long so people can read it
HIGH_GATE_DB = 18.0  # above this, soft taps are missed
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
FONTS = {}

# Ensure metadata directory exists
Path("metadata/").mkdir(exist_ok=True)
for stale in Path("metadata/").glob("*.tmp"):  # half-written saves left by a crash
    if time.time() - stale.stat().st_mtime > 3600:
        stale.unlink(missing_ok=True)


# ——— Device Persistence Helpers ———


def save_metadata():
    with open("metadata/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def load_metadata():
    defaults = {
        "default_device": "",
        "data_classes": [],
        "model_type": DEFAULT_MODEL_TYPE,
        "smooth_frames": 5,  # non-Neutral frames in a row before a tap is shown
        "min_margin": 0.5,  # a tap the model is less sure about than this is shown as Neutral
        "i1_class": "",  # trained class used as Interaction#1 (empty: the class named "bottle")
        "i2_class": "",  # trained class used as Interaction#2 (empty: the class named "bowl")
        "last_model_path": "",  # loaded again at the next start
    }
    try:
        with open("metadata/metadata.json", "r") as f:
            return defaults | json.load(f)
    except FileNotFoundError:
        return defaults


metadata = load_metadata()
if metadata["model_type"] not in MODEL_OPTIONS:
    metadata["model_type"] = DEFAULT_MODEL_TYPE

# ——— Global State ———

PLOT_Y_MIN = 20_000  # smallest y range of the spectrum plot (the quiet room fills part of it)
PLOT_Y_DECAY = 0.5 ** (1 / 88)  # per fresh frame: the y range halves in about 2 s after a loud sound
plot_ymax = PLOT_Y_MIN

audio = None
ear = None  # StreamAnalyzer instance or None
default_device = None

device_list = []  # Available input device names
device_indices_by_name = {}  # {device_name: system audio device index}

collected_data = {}  # { class_name: [data]}
record_val = None  # Currently selected class
is_recording = False  # Whether training data is currently being collected

model = None  # Loaded/trained model
infer_classes = []

DATASET_PATH = Path("metadata/dataset.pkl")  # recorded frames, kept across restarts
CALIBRATION_PATH = Path("metadata/calibration.pkl")  # quiet-room noise floor, gate and frames
LAST_MODEL_PATH = Path("metadata/last_trained.model")  # every trained model is also saved here
RESULTS_DIR = Path("results")  # one trial log (CSV) per app run that ran trials
CALIBRATION_FRAMES = 130  # about 3 s of FFT frames
EXIT_FRAMES = 3  # Neutral frames in a row before a shown tap goes back to Neutral
HOLD_FRAMES = 18  # a shown tap stays up at least this long (about 0.4 s)

# Loudness gate (set by Calibrate Quiet): frames whose ring level is below gate_db are Neutral.
gate_db = None  # ring level (dB) that a frame must reach to count as a possible tap
floor_db = None  # per-bin quiet-room spectrum in dB
calibration_frames = []  # the quiet-room frames, also used as Neutral training data
calibrating = None  # list of frames while Calibrate Quiet is collecting, else None
last_fft_count = -1  # only record/predict when the FFT has actually updated
last_freqs = None  # frequency of each FFT bin, from the latest frame
skipped_data = {}  # saved recordings not loaded (other mic settings or unknown class); written back on save
save_lock = threading.Lock()
save_version = 0  # increases with every save request; an older background save never overwrites a newer one
saved_versions = {}  # path -> version last written
inference = None  # (model, classes, smoother), always replaced together

# Screen and trial state (main thread only; worker threads hand work over with on_main)
main_jobs = queue.SimpleQueue()
ui_mode = None
role_map = T.RoleMap.from_classes([])  # which class is Neutral / Interaction#1 / Interaction#2
session = None  # trial.TrialSession: participants, trials, tally and CSV log
result_fresh = False  # the banner shows the last trial's result until the next start
counts = {T.I1: 0, T.I2: 0}  # requests shown on the product view
history = []  # (time, role), newest first
request_until = 0.0
last_role = T.NEUTRAL
live_cache = {}
ui_cache = {}  # last value written to each slow-changing widget
meter_peak, meter_t = 0.0, 0.0
plot_bins = {}  # FFT length -> number of bins plotted (up to PLOT_MAX_HZ)
last_status = None
last_error = None

# ——— Initialization ———


def init_audio():
    """Find the input devices and open the remembered (or first) one."""
    global audio, default_device, ear
    audio = pyaudio.PyAudio()
    for idx in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(idx)
        if info.get("maxInputChannels", 0) > 0:
            device_list.append(info["name"])
            device_indices_by_name[info["name"]] = idx

    default_device = device_list[0] if device_list else None
    laptop_mic = next((name for name in device_list if "MacBook" in name and "Microphone" in name), None)
    if laptop_mic:
        default_device = laptop_mic
    saved = metadata["default_device"]
    if saved in device_list:
        default_device = saved

    ear = get_ear(device=device_indices_by_name[default_device]) if default_device else None


def on_main(fn, *args):
    """Run fn(*args) in the main loop. Worker threads use this instead of changing the screen,
    the live model or the trial state themselves."""
    main_jobs.put((fn, args))


def run_main_jobs():
    while True:
        try:
            fn, args = main_jobs.get_nowait()
        except queue.Empty:
            return
        try:
            fn(*args)
        except Exception as e:
            show_status(f"Error: {e}")


def set_if_changed(tag, value, **config):
    """set_value / configure_item only when something changed (cheap to call every frame)."""
    key = (tag, value, tuple(sorted(config.items())))
    if ui_cache.get(tag) == key:
        return
    ui_cache[tag] = key
    if value is not None:
        dpg.set_value(tag, value)
    if config:
        dpg.configure_item(tag, **config)


def theme_if_changed(tag, theme):
    if ui_cache.get(("theme", tag)) != theme:
        ui_cache[("theme", tag)] = theme
        dpg.bind_item_theme(tag, theme)


# data management


def publish_model(new_model, new_classes, message=None):
    """Make a trained or loaded model live (main thread). Model, classes and smoother change in one
    step, so the frame loop never pairs a model with the wrong class list."""
    global model, infer_classes, inference
    new_classes = [str(c) for c in new_classes]
    new_smoother = LabelSmoother(neutral_class(new_classes), metadata["smooth_frames"], EXIT_FRAMES, HOLD_FRAMES)
    model, infer_classes = new_model, new_classes
    inference = (new_model, new_classes, new_smoother)
    live_cache.clear()
    update_roles()
    dpg.set_value("model_classes_text", "Model classes: " + ", ".join(new_classes))
    if message:
        show_status(message)
    if ready_to_predict():
        set_mode("DEMO")
    else:
        set_mode("SETUP")
        show_status((message + " " if message else "") + "Press Calibrate Quiet before the demo.")


def ready_to_predict():
    if inference is None:
        return False
    needs_gate = getattr(inference[0], "trained_with_gate_", False)
    return not needs_gate or (floor_db is not None and (last_freqs is None or len(floor_db) == len(last_freqs)))


def show_status(text):
    """One-line message for the user at the bottom of the window (also printed to the console)."""
    global last_status
    if text == last_status:  # messages repeated every frame are shown and printed once
        return
    last_status = text
    print(text)
    if dpg.does_item_exist("status_text"):
        dpg.set_value("status_text", text)


def format_collected_data(collected_data):
    # turns collected data into arrays for training
    x = []
    y = []
    class_names = []

    for i, (key, val) in enumerate(collected_data.items()):
        x.extend(val)
        y.extend([i for _ in range(len(val))])
        class_names.append(key)

    x = np.array(x)
    y = np.array(y)

    return x, y, class_names


def write_pickle_atomic(path, data, version=None):
    """Write to a temporary file first, so a failed save never leaves a broken file behind.
    A save with an older `version` than one already written is skipped."""
    with save_lock:
        if version is not None:
            if version < saved_versions.get(path, 0):
                return
            saved_versions[path] = version
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            replace_with_retry(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def replace_with_retry(tmp, path):
    for attempt in range(5):  # Windows: antivirus or sync tools can hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def frames_array(frames):
    """Stack frames into one array, keeping only the most common length (a device switch can mix them)."""
    lengths = [len(f) for f in frames]
    common = max(set(lengths), key=lengths.count)
    return np.asarray([f for f in frames if len(f) == common], dtype=np.float32)


def save_dataset(background=False):
    """Keep the recorded frames across restarts. With background=True the slow part (stacking
    and writing tens of MB) runs in a thread, so the window and spectrum never stall."""
    global save_version
    save_version += 1
    snapshot = {k: list(v) for k, v in collected_data.items()}  # cheap copy of the lists
    skipped = dict(skipped_data)
    if background:
        threading.Thread(target=write_dataset, args=(snapshot, skipped, save_version), daemon=True).start()
    else:
        write_dataset(snapshot, skipped, save_version)


def write_dataset(snapshot, skipped, version):
    try:
        # recordings that were not loaded (other mic settings, or a class not in the list) stay on disk
        data = {cls: frames for cls, frames in skipped.items() if cls not in snapshot}
        for cls, frames in snapshot.items():
            if frames:
                data[cls] = frames_array(frames)
            elif cls in skipped:
                data[cls] = skipped[cls]
            else:
                data[cls] = np.zeros((0, 0), dtype=np.float32)
        write_pickle_atomic(DATASET_PATH, data, version)
    except Exception as e:
        show_status(f"Could not save recordings: {e}")


def load_dataset():
    """Reload frames recorded in an earlier session, if they match this microphone's settings.
    Returns a message for the user, or None."""
    if not DATASET_PATH.exists():
        return None
    try:
        with open(DATASET_PATH, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("unexpected file format")
    except Exception as e:
        # keep the unreadable file instead of overwriting it with empty recordings later
        bad = DATASET_PATH.with_name("dataset.bad.pkl")
        try:
            os.replace(DATASET_PATH, bad)
            return f"Could not read the saved recordings ({e}); the file was kept as {bad.name}."
        except OSError:
            return f"Could not read the saved recordings ({e})."
    n_features = len(get_audio_features(ear)[1])
    refused = []
    for cls, frames in data.items():
        try:
            frames = np.asarray(frames, dtype=np.float32)
        except Exception:
            refused.append(str(cls))
            continue
        if cls not in collected_data:
            skipped_data[cls] = frames  # a class no longer in the list: keep it on disk
            continue
        if len(frames) == 0:
            continue
        if frames.ndim != 2 or frames.shape[1] != n_features:
            skipped_data[cls] = frames
            refused.append(str(cls))
            continue
        collected_data[cls] = list(frames)
        set_class_count(cls, len(frames))
    if refused:
        return (
            f"Saved recordings for {', '.join(refused)} came from different microphone settings and were not "
            "loaded; they stay on disk until you record new frames for that class."
        )
    return None


def save_calibration():
    try:
        write_pickle_atomic(
            CALIBRATION_PATH,
            {"floor_db": floor_db, "gate_db": gate_db, "frames": np.asarray(calibration_frames, dtype=np.float32)},
        )
    except Exception as e:
        show_status(f"Could not save the calibration: {e}")


def load_calibration():
    """Restore the last Calibrate Quiet result, if it matches this microphone's settings.
    Returns a message for the user."""
    global gate_db, floor_db, calibration_frames
    if not CALIBRATION_PATH.exists():
        return "Press Calibrate Quiet first (3 s of quiet)."
    try:
        with open(CALIBRATION_PATH, "rb") as f:
            data = pickle.load(f)
        floor = np.asarray(data["floor_db"], dtype=np.float64)
        gate = float(data["gate_db"])
        frames = list(data["frames"])
    except Exception as e:
        return f"Could not read the saved calibration ({e}): press Calibrate Quiet."
    if len(floor) != len(get_audio_features(ear)[1]):
        return "The saved calibration is for different microphone settings: press Calibrate Quiet."
    floor_db, gate_db, calibration_frames = floor, gate, frames
    dpg.set_value("gate_input", gate_db)
    return f"Gate {gate_db:.1f} dB from the last calibration. Press Calibrate Quiet if the room changed."


def clear_calibration(message):
    """Forget the calibration (it belongs to another microphone or setting)."""
    global gate_db, floor_db, calibration_frames
    gate_db, floor_db, calibration_frames = None, None, []
    if dpg.does_item_exist("gate_input"):
        dpg.set_value("gate_input", 0.0)
    settings_changed()
    show_status(message)


def save_last_model(new_model, new_classes):
    """Keep every trained model on disk so the next start can load it (runs in the training thread)."""
    fd, tmp = tempfile.mkstemp(dir=LAST_MODEL_PATH.parent, suffix=".tmp")
    os.close(fd)
    try:
        save_model(new_model, new_classes, Path(tmp))
        replace_with_retry(tmp, LAST_MODEL_PATH)
        on_main(remember_model, str(LAST_MODEL_PATH))
    except Exception as e:
        show_status(f"Could not keep a copy of the trained model: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def remember_model(path):
    metadata["last_model_path"] = str(path)
    save_metadata()


def load_model_file(path, note):
    """Load a .model file in a worker thread and make it live."""
    def _background_loading():
        try:
            new_model, new_classes = load_model(path)
            on_main(publish_model, new_model, new_classes, f"Loaded {path.name}{note}.")
            on_main(remember_model, str(path))
        except Exception as e:
            show_status(f"Loading {path.name} failed ({e}). Continuing with the current model.")

    threading.Thread(target=_background_loading, daemon=True).start()


def update_recording_ui():
    """Keep the recording button and status text in sync with state."""
    if not dpg.does_item_exist("record_button"):
        return

    if is_recording:
        if record_val == neutral_class(list(collected_data)):
            note = (
                " (make everyday sounds: talk, type, tap the table, handle the objects without making"
                " them ring; quiet moments are kept too)"
            )
        elif gate_db is None:
            note = " (no gate yet, so quiet gaps are recorded too: press Calibrate Quiet first)"
        else:
            note = " (only frames louder than the gate are kept)"
        dpg.configure_item("record_button", label="Stop Recording")
        dpg.bind_item_theme("record_button", "th_record_on")
        dpg.set_value("recording_status", f"Collecting data: {record_val}{note}")
        dpg.configure_item("recording_status", color=C["ok_text"])
        dpg.set_value("rec_chip", f"REC {record_val}")
    else:
        dpg.configure_item("record_button", label="Record Data")
        dpg.bind_item_theme("record_button", "th_record_idle")
        dpg.set_value("recording_status", f"Ready: {record_val}" if record_val else "Select a class first")
        dpg.configure_item("recording_status", color=C["muted"])
        dpg.set_value("rec_chip", "")


def set_recording(recording):
    global is_recording
    was_recording = is_recording
    is_recording = bool(recording and record_val)
    if was_recording and not is_recording:
        save_dataset(background=True)
    update_recording_ui()


# ——— Callback Definitions ———
def device_select_callback(s, a, u):
    """Handle user selecting a new audio device."""
    device_name = a
    metadata["default_device"] = device_name
    save_metadata()
    global ear
    if session is not None:
        handle_trial_event(session.abort("microphone changed", time.monotonic()))
    if ear is not None:
        try:
            ear.stream_reader.terminate()  # release the old stream before opening the new one
        except Exception as e:
            print(f"Could not close the previous audio stream: {e}")
    ear = get_ear(device=device_indices_by_name[device_name])
    if floor_db is not None:
        clear_calibration("Microphone changed: press Calibrate Quiet before recording or predicting.")


def model_select_callback(s, a, u):
    metadata["model_type"] = a
    save_metadata()


# Class management callbacks


def add_class_callback():
    cls = dpg.get_value("class_input").strip()
    add_class(cls)


def add_class(cls):
    if not cls or cls in collected_data:
        dpg.set_value("class_input", "")
        return
    collected_data[cls] = []
    with dpg.table_row(parent="class_table", tag=f"grp_{cls}"):
        dpg.add_text(cls, tag=f"name_{cls}")
        dpg.add_text("0", tag=f"cls_{cls}")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select", callback=select_class_button, user_data=cls)
            dpg.add_button(label="Clear", callback=clear_class_button, user_data=cls)
            dpg.add_button(label="Delete", tag=f"del_{cls}", callback=delete_class_button, user_data=cls)
            dpg.bind_item_theme(f"del_{cls}", "th_danger")
    dpg.set_value("class_input", "")
    refresh_class_colors()

    # updates the metadata
    metadata["data_classes"] = list(collected_data.keys())
    save_metadata()


def set_class_count(cls, n):
    if dpg.does_item_exist(f"cls_{cls}"):
        dpg.set_value(f"cls_{cls}", f"{n:,}")


def select_class_button(sender, app_data, user_data):
    select_class_callback(user_data)


def clear_class_button(sender, app_data, user_data):
    clear_class(user_data)


def delete_class_button(sender, app_data, user_data):
    delete_class(user_data)


def select_class_callback(cls):
    """Switch recording target to the chosen class."""
    global record_val
    record_val = cls
    dpg.set_value("selected_class", f"Selected: {cls}")
    dpg.configure_item("record_button", enabled=True)
    set_recording(False)


def record_button_callback():
    if not record_val:
        set_recording(False)
        return
    set_recording(not is_recording)


def clear_class(cls):
    collected_data[cls] = []
    skipped_data.pop(cls, None)
    set_class_count(cls, 0)
    save_dataset(background=True)


def clear_all_classes():
    for k in list(collected_data.keys()):
        clear_class(k)


def delete_class(cls):
    global record_val
    collected_data.pop(cls, None)
    skipped_data.pop(cls, None)
    dpg.delete_item(f"grp_{cls}")

    if cls == record_val:
        record_val = None
        dpg.set_value("selected_class", f"Selected: {record_val}")
        dpg.configure_item("record_button", enabled=False)
        set_recording(False)

    # updates the metadata
    metadata["data_classes"] = list(collected_data.keys())
    save_metadata()
    save_dataset(background=True)


def delete_all_classes():
    for k in list(collected_data.keys()):
        delete_class(k)


def refresh_class_colors():
    """Class names in the table take their role's color (Neutral grey, bottle blue, bowl amber)."""
    rm = role_map if role_map.classes else T.RoleMap.from_classes(
        list(collected_data), metadata["i1_class"] or None, metadata["i2_class"] or None)
    for cls in collected_data:
        kind, role = rm.role_of(cls)
        if dpg.does_item_exist(f"name_{cls}"):
            dpg.configure_item(f"name_{cls}", color=ROLE_COLOR.get(role, C["text"]) if kind == T.ROLE else C["text"])


# Mode switching
def set_mode(selected):
    """Show one of the SETUP / TRAIN / DEMO screens (the spectrum plot moves along)."""
    global ui_mode
    if selected == ui_mode:
        return
    if ui_mode == "TRAIN":
        set_recording(False)  # never keep recording training data on another screen
    if ui_mode == "DEMO" and session is not None:
        handle_trial_event(session.abort("left the DEMO screen", time.monotonic()))
    ui_mode = selected
    for m in MODES:
        dpg.configure_item(f"page_{m}", show=(m == selected))
        dpg.bind_item_theme(f"mode_btn_{m}", "th_seg_on" if m == selected else "th_seg_off")
    dpg.move_item("fft_group", parent=f"spec_slot_{selected.lower()}")
    dpg.configure_item("fft_plot", height=170 if selected == "DEMO" else -1)


def mode_button_callback(sender, app_data, user_data):
    set_mode(user_data)


# Training/inference buttons
def training_data():
    """A copy of the recordings to train on, plus notes for the user.

    With a gate, tap classes keep only frames at or above the gate (so quiet gaps recorded
    before calibrating are dropped too). Neutral keeps its loud frames (talking, typing...)
    plus some quiet-room frames, including the calibration frames, so room noise that gets
    past the gate still reads as Neutral; the quiet share is capped so it does not outweigh
    the everyday sounds that matter.
    """
    data = {k: list(v) for k, v in collected_data.items()}
    neutral = neutral_class(list(data))
    notes, quiet_dropped, other_mic = [], 0, 0
    if gate_db is not None and floor_db is not None:
        n = len(floor_db)
        for cls in data:
            other_mic += sum(1 for f in data[cls] if len(f) != n)
            frames = [f for f in data[cls] if len(f) == n]
            levels = [frame_level_db(f, last_freqs, floor_db) for f in frames]
            loud = [f for f, lvl in zip(frames, levels) if lvl >= gate_db]
            if cls == neutral:
                quiet = [f for f, lvl in zip(frames, levels) if lvl < gate_db] + list(calibration_frames)
                keep = max(130, len(loud) // 2)
                step = max(1, len(quiet) // keep)
                data[cls] = loud + quiet[::step][:keep]
            else:
                quiet_dropped += len(frames) - len(loud)
                data[cls] = loud
        if quiet_dropped:
            notes.append(f"{quiet_dropped} quiet frames were left out of the tap classes.")
        if other_mic:
            notes.append(f"{other_mic} frames from other microphone settings were skipped (clear and re-record them).")
    recorded = {cls for cls, frames in data.items() if frames and (cls != neutral or collected_data.get(neutral))}
    return data, neutral, recorded, notes


def train_model_callback():
    set_recording(False)
    data, neutral, recorded, notes = training_data()
    # makes sure theres some data to even train with (2 class min)
    if len(recorded) < 2:
        show_status("Record at least two classes (taps louder than the gate) before training. " + " ".join(notes))
        return
    if len({len(frame) for val in data.values() for frame in val}) > 1:
        show_status("Recordings come from different microphone settings: clear them and record again.")
        return
    for cls, frames in data.items():
        if cls != neutral and collected_data.get(cls) and not frames:
            notes.append(f"All '{cls}' frames were below the gate: tap louder or lower the gate.")
    if not collected_data.get(neutral):
        notes.append(f"Tip: record a '{neutral}' class (talking, typing, table taps), or loud everyday sounds may count as taps.")

    gated = gate_db is not None and floor_db is not None
    model_type = dpg.get_value("model_combo")
    dpg.configure_item("train_button", enabled=False)
    dpg.configure_item("training_status_text", show=True)
    show_status("Training...")

    def _background_training():
        try:
            X, y, class_names = format_collected_data(data)
            X = preprocess_data(X)
            new_model, new_classes = train_model(X, y, class_names, model_type)
            new_model.trained_with_gate_ = gated  # saved with the model; it then needs a gate to predict
            counts_text = ", ".join(f"{k} {len(v)}" for k, v in data.items())
            message = " ".join([f"Trained on {counts_text} frames (Neutral class: {neutral})."] + notes)
            on_main(publish_model, new_model, new_classes, message)
            save_last_model(new_model, new_classes)
        except Exception as e:
            show_status(f"Training failed: {e}")
        finally:
            on_main(training_finished)

    threading.Thread(target=_background_training, daemon=True).start()


def training_finished():
    dpg.configure_item("train_button", enabled=True)
    dpg.configure_item("training_status_text", show=False)


def calibrate_callback():
    """Measure ~3 s of the quiet room (fan, hum) and set the loudness gate above it."""
    global calibrating
    set_recording(False)
    calibrating = []
    dpg.configure_item("calib_progress", show=True)
    dpg.set_value("calib_progress", 0.0)
    show_status("Calibrating: keep the room quiet for 3 seconds...")


def finish_calibration():
    global gate_db, floor_db, calibration_frames, calibrating
    frames, calibrating = calibrating, None
    floor_db, gate_db, spread = calibrate_noise(frames, last_freqs)
    calibration_frames = frames
    save_calibration()
    dpg.set_value("gate_input", gate_db)
    dpg.configure_item("calib_progress", show=False)
    settings_changed()
    quiet_level = gate_db - GATE_MARGIN_DB
    message = (
        f"Gate set to {gate_db:.1f} dB (the quiet room reads about {quiet_level:.1f}). It applies now; "
        "train again so your recordings are re-filtered with it."
    )
    if spread > 6:
        message += f" The room was not steady while calibrating ({spread:.0f} dB spread): consider calibrating again."
    show_status(message)


def gate_input_callback(s, a, u):
    global gate_db
    if floor_db is None:
        dpg.set_value("gate_input", 0.0)
        show_status("Press Calibrate Quiet first.")
        return
    gate_db = float(a)
    save_calibration()
    settings_changed()


def smooth_input_callback(s, a, u):
    frames = max(1, int(a))
    metadata["smooth_frames"] = frames
    save_metadata()
    if inference is not None:
        inference[2].enter_frames = frames
    settings_changed()


def margin_input_callback(s, a, u):
    metadata["min_margin"] = max(0.0, float(a))
    save_metadata()
    settings_changed()


def trial_settings():
    """The recognition settings a trial ran with (logged; a change during a trial makes it INVALID)."""
    return {
        "gate_db": None if gate_db is None else round(float(gate_db), 2),
        "frames_to_switch": int(metadata["smooth_frames"]),
        "sureness": round(float(metadata["min_margin"]), 3),
    }


def settings_changed():
    if session is not None:
        session.set_settings(trial_settings(), time.monotonic())


def load_model_callback():
    dpg.configure_item("load_model_dialog", show=True)


def save_model_callback():
    dpg.configure_item("save_model_dialog", show=True)


# File dialog callbacks (these run on Dear PyGui's own thread: they only start work and hand results over)
def load_model_dialog_callback(s, a, u):
    path = a.get("file_path_name", None)
    if path:
        load_model_file(Path(path), "")


def save_model_dialog_callback(s, a, u):
    path = a.get("file_path_name", None)
    snapshot = inference  # model and classes that belong together

    def _background_saving():
        p = Path(path)
        try:
            save_model(snapshot[0], snapshot[1], p)
            on_main(remember_model, str(p))
            show_status(f"Saved {p.name}.")
        except Exception as e:
            show_status(f"Saving failed: {e}")

    if snapshot is None:
        show_status("Train or load a model before saving.")
        return
    if path:
        threading.Thread(target=_background_saving, daemon=True).start()


# ——— Roles: which trained class is Neutral, Interaction#1 and Interaction#2 ———


def update_roles():
    """Interaction#1/#2 are the classes chosen in SETUP, else the classes named bottle / bowl.
    Neutral is the class the smoother treats as Neutral."""
    global role_map
    classes = infer_classes

    def chosen(key):
        return metadata[key] if metadata[key] in classes else None

    neutral = inference[2].neutral if inference is not None else neutral_class(classes)
    role_map = T.RoleMap.from_classes(classes, chosen("i1_class"), chosen("i2_class"), neutral=neutral)
    for tag, value in (("i1_combo", role_map.i1), ("i2_combo", role_map.i2)):
        dpg.configure_item(tag, items=classes)
        dpg.set_value(tag, value or "")
    if session is not None:
        session.set_role_map(role_map, time.monotonic())
    refresh_class_colors()
    refresh_trial_panel()


def role_combo_callback(sender, app_data, user_data):
    metadata[user_data] = app_data
    save_metadata()
    update_roles()


def role_name(role):
    """Display name of a trial role: Neutral / Bottle / Bowl (or the class name of an unmapped label)."""
    if role in PRODUCT:
        return PRODUCT[role]["name"]
    m = re.fullmatch(r"Unmapped\((.*)\)", str(role))
    return m.group(1) if m else str(role)


def ui_set(labels):
    """The trial's set with screen names, roles first: {Neutral, Bottle}."""
    order = [r for r in T.ROLES if r in labels] + sorted(set(labels) - set(T.ROLES))
    return "{" + ", ".join(role_name(r) for r in order) + "}"


def type_label(trial_type):
    return trial_type.replace("I1", PRODUCT[T.I1]["name"]).replace("I2", PRODUCT[T.I2]["name"])


# ——— Live audio: gate, recording, prediction ———


def collect_calibration(amps):
    """Collects frames while Calibrate Quiet runs."""
    if calibrating is None:
        return
    calibrating.append(np.array(amps, copy=True))
    dpg.set_value("calib_progress", len(calibrating) / CALIBRATION_FRAMES)
    if len(calibrating) >= CALIBRATION_FRAMES:
        finish_calibration()


def predict_frame(amps, loud, calibrated):
    """(shown, raw, pending) for one fresh frame. Frames below the gate are Neutral, louder frames go
    to the model, the smoother hides flicker. `shown` is a short status text instead of a class
    name when the model cannot run; `pending` means a tap is building up but not shown yet."""
    live_model, classes, live_smoother = inference  # one consistent snapshot
    if getattr(live_model, "n_features_in_", len(amps)) != len(amps):
        return "model does not match this microphone: retrain", None, False
    if getattr(live_model, "trained_with_gate_", False) and not calibrated:
        # this model never saw quiet frames, so without a gate it would guess on silence
        return "press Calibrate Quiet", None, False
    raw = live_smoother.neutral
    if loud:
        neutral_index = classes.index(raw) if raw in classes else None
        X = preprocess_data(np.array([amps]))
        y_pred = int(predict_confident(live_model, X, neutral_index, metadata["min_margin"])[0])
        raw = classes[y_pred] if 0 <= y_pred < len(classes) else live_smoother.neutral
    shown = live_smoother.update(raw)
    return shown, raw, bool(live_smoother.streak)


def frame_callback():
    """Runs once per rendered frame from the main loop; does the work only when a fresh FFT frame arrived."""
    global last_fft_count, last_error, last_freqs
    now = time.monotonic()
    try:
        freqs, amps = get_audio_features(ear)

        # The GUI redraws faster than new audio arrives; only use frames with a fresh FFT.
        fft_count = ear.num_ffts if ear is not None else dpg.get_frame_count()
        if fft_count == last_fft_count or not len(amps):
            return
        last_fft_count = fft_count
        last_freqs = freqs
        show_spectrum(freqs, amps)

        # A calibration made with other microphone settings is paused, not deleted.
        calibrated = floor_db is not None and len(floor_db) == len(amps)
        if floor_db is not None and not calibrated:
            show_status("The calibration doesn't match these microphone settings: press Calibrate Quiet.")
        level = frame_level_db(amps, freqs, floor_db if calibrated else None)
        collect_calibration(amps)
        show_meter(level, calibrated, now)
        loud = not calibrated or level >= gate_db

        # recording data to collected data: once the gate is set, tap classes skip quiet frames
        # (Neutral keeps them, so the model learns that room noise is Neutral)
        if is_recording and record_val and (loud or record_val == neutral_class(list(collected_data))):
            collected_data[record_val].append(np.array(amps, copy=True))
            set_class_count(record_val, len(collected_data[record_val]))

        # live prediction, the product view and the trial recorder all use the shown (smoothed) label
        if inference is None:
            show_live(None, None)
        else:
            shown, raw, pending = predict_frame(amps, loud, calibrated)
            session.on_frame(shown, now, raw=raw, pending=pending)
            show_live(shown, raw)
            product_event(shown, now)
        last_error = None
    except Exception as e:
        if session is not None:
            session.report_error(str(e), now)  # a running trial becomes INVALID: a frame was lost
        if str(e) != last_error:  # report each new problem once instead of every frame
            last_error = str(e)
            show_status(f"Error: {e}")


def show_spectrum(freqs, amps):
    """Plot FFT amplitude against frequency (up to PLOT_MAX_HZ, where the objects ring)."""
    global plot_ymax
    n = plot_bins.get(len(freqs))
    if n is None:
        n = max(2, min(len(freqs), int(np.searchsorted(np.asarray(freqs), PLOT_MAX_HZ)) + 1))
        plot_bins.clear()
        plot_bins[len(freqs)] = n
        dpg.set_axis_limits("x_axis", 0, float(freqs[n - 1]))
    f = np.asarray(freqs[:n], dtype=float)
    a = np.asarray(amps[:n], dtype=float)
    dpg.set_value("fft_series", [f.tolist(), a.tolist()])
    # The y axis jumps up to fit a tap and then falls back slowly (halving in about 2 s), so both the
    # quiet room and a ring are readable on the built-in microphone.
    plot_ymax = max(PLOT_Y_MIN, 1.15 * float(a.max()), plot_ymax * PLOT_Y_DECAY)
    dpg.set_axis_limits("y_axis", 0, plot_ymax)
    peak = int(np.argmax(a))
    dpg.set_value("fft_status", f"Spectrum up to {f[-1] / 1000:.0f} kHz.  Peak {f[peak]:.0f} Hz, amplitude {a[peak]:.3g}")


def show_meter(level, calibrated, now):
    """Ring-level bar: bright above the gate, grey below it; the yellow tick is the gate."""
    global meter_peak, meter_t
    top = METER_TOP_DB if calibrated else 120.0  # without calibration the level is the overall level
    meter_peak = max(level, meter_peak - PEAK_FALL_DB_S * (now - meter_t))
    meter_t = now

    def x(v):
        return max(1.0, min(METER_W, METER_W * v / top))

    above = calibrated and level >= gate_db
    dpg.configure_item("meter_fill", pmax=(x(level), METER_H + 4),
                       fill=(*C["text"], 230) if above else (*C["neutral"], 150))
    dpg.configure_item("meter_peak", p1=(x(meter_peak), 2), p2=(x(meter_peak), METER_H + 6))
    dpg.set_value("level_text", f"{level:5.1f} dB")
    if calibrated:
        set_if_changed("meter_gate", None, p1=(x(gate_db), 0), p2=(x(gate_db), METER_H + 8), show=True)
        note = f"gate {gate_db:.1f} dB"
        if gate_db > HIGH_GATE_DB:
            note += "  (high: soft taps are missed, recalibrate)"
        set_if_changed("gate_text", note, color=C["warn_text"] if gate_db > HIGH_GATE_DB else C["muted"])
    else:
        set_if_changed("meter_gate", None, show=False)
        set_if_changed("gate_text", "gate not set: press Calibrate Quiet (SETUP)", color=C["warn_text"])


def show_live(shown, raw):
    """The big NOW HEARING card. Touches the widgets only when the shown label changes."""
    state = ("none",) if inference is None else role_map.role_of(shown) + (shown,)
    if live_cache.get("state") != state:
        live_cache["state"] = state
        theme = "th_hero_n"
        if inference is None:
            big, action, color = "No model", "Train one in TRAIN, or Load Model", C["warn_text"]
        else:
            kind, role, _ = state
            if kind == T.ROLE:
                big, action, color = PRODUCT[role]["name"], PRODUCT[role]["request"], ROLE_COLOR[role]
                theme = f"th_hero_{ROLE_TAG[role]}"
            elif kind == T.UNMAPPED:
                big, action, color = str(shown), "a class with no bakeoff role", C["text"]
            else:
                big, action, color = "Not ready", str(shown or "no label"), C["warn_text"]
        dpg.set_value("pred_text", big)
        dpg.configure_item("pred_text", color=color)
        dpg.set_value("hero_action", action)
        dpg.configure_item("hero_action", color=color if state[0] != T.ROLE or state[1] != T.NEUTRAL else C["muted"])
        dpg.configure_item("hero_dot", fill=(*color, 255), color=(*color, 255))
        dpg.bind_item_theme("hero", theme)
    raw_text = f"this frame: {raw}" if raw is not None else ""
    if live_cache.get("raw") != raw_text:
        live_cache["raw"] = raw_text
        dpg.set_value("raw_pred_text", raw_text)


# ——— Product view (DEMO, left) ———


def product_event(shown, now):
    """One request per change of the shown label to the bottle or the bowl (never per frame)."""
    global last_role, request_until
    kind, role = role_map.role_of(shown)
    role = role if kind == T.ROLE else None
    if role in (T.I1, T.I2) and role != last_role and ui_mode == "DEMO":
        counts[role] += 1
        history.insert(0, (time.strftime("%H:%M:%S"), role))
        del history[LOG_ROWS:]
        request_until = now + REQUEST_SHOW_S
        dpg.set_value("request_text", f"{PRODUCT[role]['request']} - {time.strftime('%H:%M')}")
        dpg.configure_item("request_text", color=ROLE_COLOR[role])
        for r in (T.I1, T.I2):
            theme_if_changed(f"tile_{ROLE_TAG[r]}", f"th_tile_{ROLE_TAG[r]}" + ("_flash" if r == role else ""))
        refresh_product()
    last_role = role


def refresh_product():
    for r in (T.I1, T.I2):
        dpg.set_value(f"count_{ROLE_TAG[r]}", str(counts[r]))
    for k in range(LOG_ROWS):
        t, r = history[k] if k < len(history) else ("", None)
        dpg.set_value(f"log_t_{k}", t)
        dpg.set_value(f"log_l_{k}", PRODUCT[r]["name"] if r else "")
        dpg.set_value(f"log_a_{k}", PRODUCT[r]["request"] if r else "")
        dpg.configure_item(f"log_l_{k}", color=ROLE_COLOR.get(r, C["text"]))


def reset_counts_callback():
    global request_until
    counts[T.I1] = counts[T.I2] = 0
    history.clear()
    request_until = 0.0
    refresh_product()


# ——— Trial recorder (DEMO, right) ———


class KeyEdge:
    """Polled once per frame in the main loop: one press per physical press (auto-repeat never fires
    twice), and it runs on the main thread (Dear PyGui's key handlers run on another thread)."""

    def __init__(self, *keys):
        self.keys, self.down = keys, False

    def poll(self):
        down = any(dpg.is_key_down(k) for k in self.keys)
        pressed, released = down and not self.down, self.down and not down
        self.down = down
        return pressed, released


TEXT_INPUTS = ("next_pid_input", "class_input", "gate_input", "smooth_input", "margin_input", "void_reason")
MODALS = ("load_model_dialog", "save_model_dialog", "confirm_next", "void_dialog")


def keyboard_busy():
    """True while someone types in a box or a dialog is open: SPACE then belongs to that widget."""
    if any(dpg.does_item_exist(t) and dpg.is_item_active(t) for t in TEXT_INPUTS):
        return True
    return any(dpg.does_item_exist(t) and dpg.is_item_shown(t) for t in MODALS)


def shift_down():
    return dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)


def trial_press(now, override=False, key=True):
    """SPACE (key=True) or a click on the banner: start a trial, or stop the running one."""
    if ui_mode != "DEMO" or session is None:
        return
    if inference is None and session.state != T.RECORDING:
        show_status("Train or load a model before running trials.")
        return
    if key:
        ev = session.space_down(now, override=override, text_input_active=keyboard_busy())
    else:
        ev = session.press(now, override=override)
    handle_trial_event(ev)


def handle_trial_event(ev):
    global result_fresh
    if ev.action == "started":
        result_fresh = False
        print(ev.message)
    elif ev.action in ("stopped", "aborted") and ev.result is not None:
        res = ev.result
        # the bakeoff asks the code to output the set when the trial stops
        print(f"Trial #{res.trial_no} {res.trial_type}: set = {res.set_text} = {ui_set(res.labels)} -> {res.verdict}"
              + (f" ({res.reason})" if res.reason and res.verdict != T.SET_ONLY else ""))
        result_fresh = True
        suggested = session.tally.suggested_type()
        if suggested and suggested != session.selected_type:
            session.select_type(suggested)  # the next trial type is selected for the experimenter
    elif ev.action == "refused":
        show_status(f"Trial not started: {ev.message}")
    if ev.action in ("started", "stopped", "aborted", "selected", "voided", "participant"):
        refresh_trial_panel()


def banner_callback():
    trial_press(time.monotonic(), key=False)


def override_callback():
    trial_press(time.monotonic(), override=True, key=False)


def trial_type_callback(sender, app_data, user_data):
    ev = session.select_type(user_data)
    if not ev.ok:
        show_status(ev.message)
    refresh_trial_panel()


def open_modal(tag, width, height):
    w, h = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
    dpg.configure_item(tag, pos=(max(0, (w - width) // 2), max(0, (h - height) // 3)), show=True)


def void_callback():
    tal = session.tally
    if session.state == T.RECORDING or tal is None or not tal.trials or tal.trials[-1].voided:
        show_status("Nothing to void: only the most recent trial of this participant can be voided, once.")
        return
    last = tal.trials[-1]
    dpg.set_value("void_text", f"Void trial #{last.trial_no} ({type_label(last.trial_type)}, {last.verdict})? "
                               "Only for an experimenter mistake, such as the wrong type selected or SPACE pressed "
                               "too early. It stays in the log marked void, and its slot opens again.")
    open_modal("void_dialog", 560, 230)


def void_confirmed_callback():
    global result_fresh
    dpg.configure_item("void_dialog", show=False)
    ev = session.void_last(dpg.get_value("void_reason"), time.monotonic())
    show_status(ev.message)
    if ev.ok:
        result_fresh = False
        session.select_type(ev.result.trial_type)  # run it again
    refresh_trial_panel()


def next_participant_callback():
    pid = dpg.get_value("next_pid_input").strip()
    old = session.tally
    if pid and old is not None and old.real_started and not old.complete:
        dpg.set_value("confirm_next_text", f"{old.participant} has {old.n_counted_total} of "
                                           f"{T.TRIALS_PER_PARTICIPANT} counted trials. Start {pid} anyway? The "
                                           "unfinished block stays in the log.")
        open_modal("confirm_next", 560, 190)
        return
    start_participant(pid, force=False)


def confirm_next_callback():
    dpg.configure_item("confirm_next", show=False)
    start_participant(dpg.get_value("next_pid_input").strip(), force=True)


def start_participant(pid, force):
    global result_fresh
    ev = session.next_participant(pid, time.monotonic(), force=force)
    show_status(ev.message)
    if ev.ok:
        result_fresh = False
        dpg.set_value("next_pid_input", next_participant_id(pid))
    refresh_trial_panel()


def next_participant_id(pid):
    """P3 -> P4; anything without a trailing number gets one."""
    m = re.search(r"(\d+)$", pid)
    return pid[: m.start()] + str(int(m.group(1)) + 1) if m else pid + "2"


def copy_numbers_callback():
    tal = session.tally
    if tal is None:
        return
    rows = tal.sheet_rows()
    dpg.set_clipboard_text("\n".join(f"{a}\t{b}\t{acc:.1f}" for _, a, b, acc in rows))
    show_status(f"Copied {tal.participant}'s numbers: paste into the Interaction#1 cell of the Positive row.")


def refresh_trial_panel():
    """Everything in the trial panel except the banner (which show_banner updates every frame)."""
    if session is None or not dpg.does_item_exist("trial_panel"):
        return
    tal = session.tally
    if tal is not None:
        dpg.set_value("participant_text", f"Participant {tal.participant}")
        extra = []
        if tal.invalid_count:
            extra.append(f"{tal.invalid_count} invalid (re-run)")
        if tal.voided_count:
            extra.append(f"{tal.voided_count} voided")
        done = "COMPLETE" if tal.complete else f"{tal.n_counted_total}/{T.TRIALS_PER_PARTICIPANT} counted"
        dpg.set_value("trial_progress_text", "   ".join([done] + extra))
    selected = session.selected_type
    for t, key in TYPE_KEY.items():
        tag = f"ttype_{key}"
        if t in T.PRACTICE_TYPES:
            name = "Neutral" if t == T.PRACTICE_NEUTRAL else role_name(T.I1 if t == T.PRACTICE_I1 else T.I2)
            done = tal is not None and tal.practice_done(t)
            label = name + ("  done" if done else "")
            enabled = tal is not None and not tal.real_started and not done
        else:
            n = tal.n_counted(t) if tal else 0
            label = f"{role_name(T.I1 if t in (T.POS_I1, T.NEG_I1) else T.I2)}  {n}/{T.TRIALS_PER_CELL}"
            enabled = tal is not None and not tal.full(t)
        dpg.configure_item(tag, label=label, enabled=enabled or t == selected)
        dpg.bind_item_theme(tag, "th_seg_on" if t == selected else "th_seg_off")
    for rk, row in zip(("pos", "neg"), tal.sheet_rows() if tal else [["", 0, 0, 0.0]] * 2):
        _, a, b, acc = row
        dpg.set_value(f"tally_{rk}_1", str(a))
        dpg.set_value(f"tally_{rk}_2", str(b))
        dpg.set_value(f"tally_{rk}_acc", f"{acc:.1f}")
    for t, (r, c) in TALLY_CELL.items():
        if t == selected:
            dpg.highlight_table_cell("tally_table", r, c, (*C["accent_fill"], 90))
        else:
            dpg.unhighlight_table_cell("tally_table", r, c)
    if tal is not None:
        total = sum(tal.successes(c) for c in T.REAL_TYPES)
        dpg.set_value("tally_overall", f"Overall {total}/{T.TRIALS_PER_PARTICIPANT}")
    dpg.set_value("trial_say", SAY.get(selected, ""))
    res = session.last_result
    if res is not None:
        text = f"Last: #{res.trial_no} {type_label(res.trial_type)}   set {res.set_text}   {res.verdict}"
        notes = []
        if res.reason and res.verdict != T.SET_ONLY:
            notes.append(res.reason)
        if res.practice:
            notes.append("practice, not counted")
        elif not res.counted:
            notes.append("not counted")
        if res.voided:
            notes.append("VOIDED: " + res.void_reason)
        dpg.set_value("trial_result", text + (f"   ({'; '.join(notes)})" if notes else "") + f"   {res.duration_s:.1f} s")
    else:
        dpg.set_value("trial_result", "")
    problems = role_map.problems()
    if inference is None:
        roles = "No model yet: train or load one."
    elif problems:
        roles = "Fix the roles in SETUP: " + "; ".join(problems)
    else:
        roles = (f"Interaction#1 = {role_name(T.I1)} (class '{role_map.i1}')    "
                 f"Interaction#2 = {role_name(T.I2)} (class '{role_map.i2}')    Neutral = class '{role_map.neutral}'")
    dpg.set_value("roles_text", roles)
    dpg.configure_item("roles_text", color=C["bad_text"] if inference is not None and problems else C["muted"])
    log = session.log
    if log is not None and log.last_error:
        dpg.set_value("trial_log_text", f"Log problem: {log.last_error} ({len(log.pending)} rows waiting; trials go on)")
        dpg.configure_item("trial_log_text", color=C["bad_text"])
    elif log is not None:
        where = log.path.as_posix() if log.path.exists() else f"{log.path.as_posix()} (created at the first trial)"
        dpg.set_value("trial_log_text", f"Every trial is logged to {where}")
        dpg.configure_item("trial_log_text", color=C["muted"])


BLOCK_TITLES = {
    "participant": "NO PARTICIPANT", "roles": "SET ROLES", "complete": "ALL 20 DONE", "practice": "PRACTICE DONE",
    "cell_full": "5 OF 5 DONE", "no_audio": "NO AUDIO", "app_error": "NOT READY", "not_neutral": "WAIT FOR NEUTRAL",
    "type": "PICK A TYPE",
}
KEEP_RESULT_FOR = ("not_neutral", "complete", "cell_full", "practice")  # blocks that leave the result on the banner


def block_text(block):
    return block[1].replace("APP ERROR: ", "").replace("NO AUDIO: ", "")


VERDICT_LOOK = {T.PASS: ("PASS", "th_banner_ok"), T.FAIL: ("FAIL", "th_banner_bad"),
                T.INVALID: ("INVALID", "th_banner_wait"), T.SET_ONLY: ("SET", "th_banner_info")}


def show_banner(now):
    """The big trial banner: READY / RECORDING / the result / why a start would be refused."""
    st = session.status(now)
    block = st["start_block"]
    override = False
    if st["state"] == T.RECORDING:
        label, theme = f"RECORDING  {st['elapsed_s']:.1f} s", "th_banner_rec"
        hint = "The participant acts now. Press SPACE when they are done (Esc cancels; it is not counted)."
    elif inference is None:
        label, theme, hint = "NO MODEL", "th_banner_wait", "Train a model in TRAIN, or Load Model."
    elif result_fresh and session.last_result is not None and (block is None or block[0] in KEEP_RESULT_FOR):
        # the last result stays up until the next start, unless something needs fixing first
        res = session.last_result
        word, theme = VERDICT_LOOK.get(res.verdict, (res.verdict, "th_banner_wait"))
        label = f"{word}  {ui_set(res.labels)}"
        nxt = f"Next: {type_label(session.selected_type)}. "
        hint = nxt + ("Read the line below aloud, then press SPACE." if block is None else block_text(block))
        override = st["can_override"]
    elif block is not None:
        label, theme = BLOCK_TITLES.get(block[0], "NOT READY"), "th_banner_wait"
        hint = block_text(block)
        override = st["can_override"]
    else:
        label, theme = f"READY  {type_label(session.selected_type)}", "th_banner_ready"
        hint = "Read the line below aloud, then press SPACE (or click here) to start."
    set_if_changed("trial_banner", None, label=label)
    theme_if_changed("trial_banner", theme)
    set_if_changed("trial_hint", hint)
    set_if_changed("override_btn", None, show=override)


# ——— Periodic screen updates ———

next_slow_tick = 0.0


def ui_tick(now):
    global request_until, next_slow_tick
    if request_until and now > request_until:
        request_until = 0.0
        dpg.set_value("request_text", PRODUCT[T.NEUTRAL]["request"])
        dpg.configure_item("request_text", color=C["muted"])
        for r in (T.I1, T.I2):
            theme_if_changed(f"tile_{ROLE_TAG[r]}", f"th_tile_{ROLE_TAG[r]}")
    if session is not None:
        show_banner(now)
    if now >= next_slow_tick:
        next_slow_tick = now + 0.5
        refresh_chips()
        refresh_checklist()


def refresh_chips():
    mic = metadata["default_device"] or default_device or "no microphone"
    set_if_changed("mic_chip", f"MIC  {mic[:34]}")
    if inference is None:
        set_if_changed("model_chip", "NO MODEL", color=C["warn_text"])
    elif role_map.problems():
        set_if_changed("model_chip", "MODEL: set roles in SETUP", color=C["bad_text"])
    else:
        set_if_changed("model_chip", "MODEL  " + " / ".join(infer_classes), color=C["muted"])


def refresh_checklist():
    def box(ok):
        return "[x]" if ok else "[  ]"

    smooth, margin = metadata["smooth_frames"], metadata["min_margin"]
    lines = [
        f"{box(ear is not None)} Microphone: {metadata['default_device'] or default_device or 'none found'}",
        f"{box(gate_db is not None)} Quiet calibration: "
        + (f"gate {gate_db:.1f} dB" if gate_db is not None else "press Calibrate Quiet"),
        f"{box(inference is not None)} Model: " + (", ".join(infer_classes) if inference else "train or load one"),
        f"{box(inference is not None and not role_map.problems())} Roles: "
        + (f"{role_map.i1} = Interaction#1, {role_map.i2} = Interaction#2" if inference and not role_map.problems()
           else "need a model with a bottle and a bowl class"),
        f"{box(smooth == 5 and abs(margin - 0.5) < 1e-9)} Frames to switch {smooth}, sureness {margin:.2f}"
        " (we tested 5 and 0.50)",
    ]
    if gate_db is not None and gate_db > HIGH_GATE_DB:
        lines.append(f"Gate {gate_db:.1f} dB is high: soft taps are missed. Recalibrate (about 11-13 dB).")
    set_if_changed("setup_checklist", "\n".join(lines))


last_layout = None


def poll_layout():
    """React to window size changes (replaces the starter's resize callback)."""
    global last_layout
    size = (dpg.get_viewport_client_width(), dpg.get_viewport_client_height())
    if size == last_layout or size[0] <= 0:
        return
    last_layout = size
    w = size[0]
    panel_w = max(420, int(w * 0.36))
    dpg.configure_item("product_panel", width=panel_w)
    tile_w = (panel_w - 3 * 10 - 6) // 2
    for key in ("i1", "i2"):
        dpg.configure_item(f"tile_{key}", width=tile_w)
    dpg.configure_item("status_text", wrap=max(300, w - 760))


# ——— Look: fonts, themes, icons ———


def load_fonts():
    # ImGui maps the size to ascent minus descent, so Segoe UI "20" looks like 15 px text (Windows 11 pt)
    spec = {"body": ("segoeui.ttf", 20), "bold": ("segoeuib.ttf", 20), "h2": ("segoeuib.ttf", 28),
            "banner": ("segoeuib.ttf", 44), "huge": ("segoeuib.ttf", 72)}
    with dpg.font_registry():
        for name, (fname, size) in spec.items():
            path = FONT_DIR / fname
            FONTS[name] = dpg.add_font(str(path), size) if path.is_file() else None
    if FONTS["body"]:
        dpg.bind_font(FONTS["body"])
    else:
        dpg.set_global_font_scale(1.25)  # no Segoe UI (not Windows): enlarge the built-in font


def font(item, name):
    f = FONTS.get(name) or FONTS.get("bold") or FONTS.get("body")
    if f:
        dpg.bind_item_font(item, f)


def mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def rgba(c, a=255):
    return (*c[:3], a)


def button_theme(tag, base, text=(255, 255, 255), hover=None, active=None):
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgba(base), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgba(hover or mix(base, (255, 255, 255), 0.12)),
                                category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgba(active or mix(base, (255, 255, 255), 0.22)),
                                category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(text), category=dpg.mvThemeCat_Core)
        with dpg.theme_component(dpg.mvButton, enabled_state=False):  # disabled buttons need their own look
            dim = rgba(mix(C["card"], base, 0.25))
            dpg.add_theme_color(dpg.mvThemeCol_Button, dim, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, dim, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, dim, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgba(C["muted"]), category=dpg.mvThemeCat_Core)


def child_theme(tag, bg, border=None):
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, rgba(bg), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, rgba(border or C["border"]), category=dpg.mvThemeCat_Core)


def build_themes():
    core = dpg.mvThemeCat_Core
    with dpg.theme(tag="th_global"):
        with dpg.theme_component(dpg.mvAll):
            for target, color in (
                (dpg.mvThemeCol_WindowBg, C["bg"]), (dpg.mvThemeCol_ChildBg, C["card"]),
                (dpg.mvThemeCol_PopupBg, C["card2"]), (dpg.mvThemeCol_Border, C["border"]),
                (dpg.mvThemeCol_FrameBg, C["frame"]), (dpg.mvThemeCol_FrameBgHovered, C["frame_hi"]),
                (dpg.mvThemeCol_FrameBgActive, C["frame_act"]), (dpg.mvThemeCol_Button, C["frame"]),
                (dpg.mvThemeCol_ButtonHovered, C["frame_hi"]), (dpg.mvThemeCol_ButtonActive, C["frame_act"]),
                (dpg.mvThemeCol_Text, C["text"]), (dpg.mvThemeCol_TextDisabled, C["muted"]),
                (dpg.mvThemeCol_CheckMark, C["accent"]), (dpg.mvThemeCol_SliderGrab, C["accent"]),
                (dpg.mvThemeCol_SliderGrabActive, C["accent"]), (dpg.mvThemeCol_Separator, C["border"]),
                (dpg.mvThemeCol_TableHeaderBg, C["card2"]), (dpg.mvThemeCol_TableBorderStrong, C["border"]),
                (dpg.mvThemeCol_TableBorderLight, mix(C["card"], C["border"], 0.6)),
                (dpg.mvThemeCol_ScrollbarBg, C["card"]), (dpg.mvThemeCol_ScrollbarGrab, C["frame_hi"]),
                (dpg.mvThemeCol_ScrollbarGrabHovered, C["frame_act"]), (dpg.mvThemeCol_ScrollbarGrabActive, C["accent"]),
                (dpg.mvThemeCol_TitleBg, C["card2"]), (dpg.mvThemeCol_TitleBgActive, C["card2"]),
                (dpg.mvThemeCol_MenuBarBg, C["card"]), (dpg.mvThemeCol_PlotHistogram, C["accent_fill"]),
                (dpg.mvThemeCol_NavHighlight, C["accent"]),
            ):
                dpg.add_theme_color(target, rgba(color), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_Header, rgba(C["accent_fill"], 110), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, rgba(C["accent_fill"], 150), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, rgba(C["accent_fill"], 200), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, (0, 0, 0, 0), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, (255, 255, 255, 8), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, rgba(C["accent_fill"], 120), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg, (0, 0, 0, 160), category=core)
            for target, x, y in (
                (dpg.mvStyleVar_WindowPadding, 10, 10), (dpg.mvStyleVar_FramePadding, 12, 7),
                (dpg.mvStyleVar_ItemSpacing, 10, 8), (dpg.mvStyleVar_ItemInnerSpacing, 8, 6),
                (dpg.mvStyleVar_CellPadding, 10, 6), (dpg.mvStyleVar_ButtonTextAlign, 0.5, 0.5),
            ):
                dpg.add_theme_style(target, x, y, category=core)
            for target, v in (
                (dpg.mvStyleVar_WindowRounding, 0), (dpg.mvStyleVar_ChildRounding, 10),
                (dpg.mvStyleVar_FrameRounding, 7), (dpg.mvStyleVar_PopupRounding, 8),
                (dpg.mvStyleVar_GrabRounding, 7), (dpg.mvStyleVar_ScrollbarRounding, 8),
                (dpg.mvStyleVar_ScrollbarSize, 14), (dpg.mvStyleVar_WindowBorderSize, 0),
                (dpg.mvStyleVar_ChildBorderSize, 1), (dpg.mvStyleVar_FrameBorderSize, 0),
            ):
                dpg.add_theme_style(target, v, category=core)
        with dpg.theme_component(dpg.mvPlot):
            plots = dpg.mvThemeCat_Plots
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg, rgba(C["card"]), category=plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, rgba(C["bg"]), category=plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, rgba(C["border"]), category=plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, (255, 255, 255, 22), category=plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisText, rgba(C["muted"]), category=plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg, rgba(C["card2"], 220), category=plots)
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, rgba(C["accent"]), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.6, category=dpg.mvThemeCat_Plots)
    dpg.bind_theme("th_global")

    for key in ("i1", "i2"):
        with dpg.theme(tag=f"th_marks_{key}"):
            with dpg.theme_component(dpg.mvInfLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, rgba(C[key], 150), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.0, category=dpg.mvThemeCat_Plots)

    button_theme("th_seg_on", C["accent_fill"])
    button_theme("th_seg_off", C["frame"], text=C["muted"])
    button_theme("th_primary", C["accent_fill"])
    button_theme("th_danger", (127, 29, 29))
    button_theme("th_record_idle", (127, 29, 29))
    button_theme("th_record_on", C["rec"])
    for name in ("ready", "rec", "ok", "bad", "wait", "info"):  # trial banner: hover = base, so it never flickers
        button_theme(f"th_banner_{name}", C[name], hover=C[name], active=mix(C[name], (0, 0, 0), 0.15))
    child_theme("th_bar", C["card"], C["card"])
    child_theme("th_flat", C["bg"], C["bg"])
    for role, key in ROLE_TAG.items():
        child_theme(f"th_hero_{key}", mix(C["card"], ROLE_COLOR[role], 0.10), mix(C["border"], ROLE_COLOR[role], 0.6))
    for key in ("i1", "i2"):
        child_theme(f"th_tile_{key}", mix(C["card"], C[key], 0.08), mix(C["border"], C[key], 0.35))
        child_theme(f"th_tile_{key}_flash", mix(C["card"], C[key], 0.40), C[key])


def draw_bottle(parent, w, h, col):
    cx = w / 2
    dpg.draw_rectangle((cx - 20, h * 0.36), (cx + 20, h - 4), fill=rgba(col, 230), color=rgba(col), rounding=10,
                       parent=parent)
    dpg.draw_quad((cx - 20, h * 0.38), (cx + 20, h * 0.38), (cx + 8, h * 0.22), (cx - 8, h * 0.22),
                  fill=rgba(col, 230), color=rgba(col), parent=parent)
    dpg.draw_rectangle((cx - 8, h * 0.10), (cx + 8, h * 0.23), fill=rgba(col, 230), color=rgba(col), parent=parent)
    dpg.draw_rectangle((cx - 10, 3), (cx + 10, h * 0.11), fill=rgba(mix(col, (255, 255, 255), 0.4)), color=rgba(col),
                       rounding=3, parent=parent)
    ring_marks(parent, cx + 30, h * 0.62, col)


def draw_bowl(parent, w, h, col):
    cx, top, rx, ry = w / 2, h * 0.45, 48, h * 0.45
    pts = [(cx + rx * cos(pi * i / 24), top + ry * sin(pi * i / 24)) for i in range(25)]
    dpg.draw_polygon(pts + [pts[0]], fill=rgba(col, 230), color=rgba(col), parent=parent)
    dpg.draw_ellipse((cx - rx, top - 7), (cx + rx, top + 7), fill=rgba(mix(col, (0, 0, 0), 0.35)), color=rgba(col),
                     parent=parent)
    dpg.draw_line((cx - 66, 8), (cx - 30, top - 4), color=rgba(C["text"]), thickness=4, parent=parent)  # spoon
    dpg.draw_ellipse((cx - 36, top - 13), (cx - 20, top - 2), fill=rgba(C["text"]), color=rgba(C["text"]),
                     parent=parent)
    ring_marks(parent, cx + rx + 12, top + 8, col)


def ring_marks(parent, x, y, col):
    for k, r in enumerate((7, 13, 19)):
        dpg.draw_bezier_quadratic((x + r * 0.2, y - r), (x + r, y), (x + r * 0.2, y + r),
                                  color=rgba(col, 220 - 60 * k), thickness=2, parent=parent)


# ——— UI Construction ———


def section(text):
    dpg.add_text(text.upper(), color=C["muted"])
    dpg.add_separator()


def build_ui():
    with dpg.window(tag="root", no_scrollbar=True):
        with dpg.child_window(tag="topbar", height=TOPBAR_H, border=False, no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_text(PRODUCT["name"], tag="app_title")
                font("app_title", "h2")
                dpg.add_spacer(width=16)
                for m in MODES:
                    dpg.add_button(label=m, tag=f"mode_btn_{m}", width=120, height=34, callback=mode_button_callback,
                                   user_data=m)
                dpg.add_spacer(width=16)
                dpg.add_text("", tag="mic_chip", color=C["muted"])
                dpg.add_text("", tag="model_chip", color=C["warn_text"])
                dpg.add_text("", tag="rec_chip", color=C["rec"])
        dpg.bind_item_theme("topbar", "th_bar")

        build_status_band()
        with dpg.child_window(tag="mode_area", height=-FOOTER_H, border=False):
            build_setup_page()
            build_train_page()
            build_demo_page()
        dpg.bind_item_theme("mode_area", "th_flat")

        with dpg.child_window(tag="footer", height=-1, border=False, no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_text("Press Calibrate Quiet first (3 s of quiet).", tag="status_text", color=C["warn_text"],
                             wrap=600)
                dpg.add_text(KEY_HINTS, tag="hint_text", color=C["muted"])
        dpg.bind_item_theme("footer", "th_flat")

    # One spectrum plot, moved between the screens with move_item
    with dpg.group(tag="fft_group", parent="spec_slot_setup"):
        dpg.add_text("Waiting for microphone data...", tag="fft_status", color=C["muted"])
        with dpg.plot(tag="fft_plot", height=-1, width=-1, no_menus=True, no_box_select=True, no_mouse_pos=True):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis", label="Frequency (Hz)")
            dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis", label="Amplitude", no_tick_labels=True)
            dpg.set_axis_limits("y_axis", 0, PLOT_Y_MIN)
            dpg.add_line_series([], [], label="spectrum", parent="y_axis", tag="fft_series")
            for role in (T.I1, T.I2):
                key = ROLE_TAG[role]
                dpg.add_inf_line_series([float(f) for f in PRODUCT[role]["ring_hz"]], parent="y_axis",
                                        tag=f"marks_{key}", label=f"{PRODUCT[role]['name'].lower()} rings")
                dpg.bind_item_theme(f"marks_{key}", f"th_marks_{key}")

    build_dialogs()


def build_status_band():
    with dpg.group(horizontal=True):
        with dpg.child_window(tag="hero", width=-(METER_W + 50), height=HERO_H, no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_text("NOW HEARING", color=C["muted"])
                dpg.add_spacer(width=12)
                dpg.add_text("", tag="raw_pred_text", color=C["muted"])
            with dpg.group(horizontal=True):
                with dpg.drawlist(width=50, height=72, tag="hero_dot_dl"):
                    dpg.draw_circle((25, 38), 17, fill=rgba(C["neutral"]), color=rgba(C["neutral"]), tag="hero_dot")
                dpg.add_text("No model", tag="pred_text", color=C["warn_text"])
                font("pred_text", "huge")
                with dpg.group():
                    dpg.add_spacer(height=22)
                    dpg.add_text("", tag="hero_action", color=C["muted"])
                    font("hero_action", "h2")
        dpg.bind_item_theme("hero", "th_hero_n")
        with dpg.child_window(tag="meter_card", width=-1, height=HERO_H, no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_text("RING LEVEL", color=C["muted"])
                dpg.add_text("--", tag="level_text")
                font("level_text", "bold")
            with dpg.drawlist(width=METER_W, height=METER_H + 8, tag="meter_dl"):
                dpg.draw_rectangle((0, 4), (METER_W, METER_H + 4), fill=rgba(C["frame"]), color=rgba(C["frame"]),
                                   rounding=6, tag="meter_bg")
                dpg.draw_rectangle((0, 4), (1, METER_H + 4), fill=rgba(C["neutral"], 150), color=(0, 0, 0, 0),
                                   rounding=6, tag="meter_fill")
                dpg.draw_line((0, 2), (0, METER_H + 6), color=rgba(C["text"], 160), thickness=2, tag="meter_peak")
                dpg.draw_line((0, 0), (0, METER_H + 8), color=rgba(C["warn_text"]), thickness=3, tag="meter_gate",
                              show=False)
            dpg.add_text("gate not set: press Calibrate Quiet (SETUP)", tag="gate_text", color=C["warn_text"])


def build_setup_page():
    with dpg.child_window(tag="page_SETUP", border=False, show=False):
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="setup_left", width=470, height=-1):
                section("Microphone")
                dpg.add_combo(items=device_list, tag="device_combo", default_value=default_device or "", width=-1,
                              callback=device_select_callback)
                dpg.add_spacer(height=4)
                section("Quiet-room calibration")
                dpg.add_button(label="Calibrate Quiet  (keep the room quiet for 3 s)", tag="calib_button", width=-1,
                               height=40, callback=calibrate_callback)
                dpg.bind_item_theme("calib_button", "th_primary")
                dpg.add_progress_bar(tag="calib_progress", default_value=0.0, overlay="keep quiet...", width=-1,
                                     show=False)
                dpg.add_input_float(tag="gate_input", label="gate dB (Enter)", default_value=0.0, step=1.0,
                                    format="%.1f", width=150, on_enter=True, callback=gate_input_callback)
                dpg.add_text("Louder frames than the gate can be taps; quieter frames are Neutral.", color=C["muted"],
                             wrap=0)
                dpg.add_spacer(height=4)
                section("Recognition")
                dpg.add_input_int(tag="smooth_input", label="frames to switch", default_value=metadata["smooth_frames"],
                                  min_value=1, max_value=10, min_clamped=True, max_clamped=True, width=150,
                                  callback=smooth_input_callback)
                dpg.add_input_float(tag="margin_input", label="sureness (Enter)", default_value=metadata["min_margin"],
                                    step=0.25, min_value=0.0, min_clamped=True, format="%.2f", width=150,
                                    on_enter=True, callback=margin_input_callback)
                dpg.add_spacer(height=4)
                section("Bakeoff roles")
                dpg.add_combo(items=[], tag="i1_combo", label=f"Interaction#1 ({PRODUCT[T.I1]['name']})", width=180,
                              callback=role_combo_callback, user_data="i1_class")
                dpg.add_combo(items=[], tag="i2_combo", label=f"Interaction#2 ({PRODUCT[T.I2]['name']})", width=180,
                              callback=role_combo_callback, user_data="i2_class")
                dpg.add_spacer(height=4)
                section("Ready for the demo?")
                dpg.add_text("", tag="setup_checklist", wrap=0)
            with dpg.child_window(tag="setup_right", width=-1, height=-1):
                dpg.add_group(tag="spec_slot_setup")


def build_train_page():
    with dpg.child_window(tag="page_TRAIN", border=False, show=False):
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="train_left", width=-480, height=-1):
                section("Classes and recorded frames")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="class_input", hint="new class name", width=-130, on_enter=True,
                                       callback=add_class_callback)
                    dpg.add_button(label="Add Class", width=-1, callback=add_class_callback)
                with dpg.table(tag="class_table", header_row=True, policy=dpg.mvTable_SizingStretchProp,
                               borders_innerH=True, borders_outerH=True, row_background=True, height=190,
                               scrollY=True):
                    dpg.add_table_column(label="Class", init_width_or_weight=1.4)
                    dpg.add_table_column(label="Frames", init_width_or_weight=0.8)
                    dpg.add_table_column(label="", init_width_or_weight=2.2, no_sort=True)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Clear All", callback=clear_all_classes)
                    dpg.add_button(label="Delete All", tag="delete_all_btn", callback=delete_all_classes)
                    dpg.bind_item_theme("delete_all_btn", "th_danger")
                dpg.add_spacer(height=4)
                dpg.add_group(tag="spec_slot_train")
            with dpg.child_window(tag="train_right", width=-1, height=-1):
                section("Record")
                dpg.add_text(f"Selected: {record_val}", tag="selected_class")
                font("selected_class", "h2")
                dpg.add_button(label="Record Data", tag="record_button", width=-1, height=56, enabled=False,
                               callback=record_button_callback)
                dpg.bind_item_theme("record_button", "th_record_idle")
                font("record_button", "h2")
                dpg.add_text("Select a class first", tag="recording_status", color=C["muted"], wrap=0)
                dpg.add_spacer(height=8)
                section("Model")
                dpg.add_combo(items=MODEL_OPTIONS, default_value=metadata["model_type"], tag="model_combo", width=-1,
                              callback=model_select_callback)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Train Model", tag="train_button", width=200, height=40,
                                   callback=train_model_callback)
                    dpg.bind_item_theme("train_button", "th_primary")
                    dpg.add_loading_indicator(tag="training_status_text", show=False, radius=2.0,
                                              color=rgba(C["accent"]), secondary_color=rgba(C["accent_fill"]))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save Model", tag="btn_save", width=200, callback=save_model_callback)
                    dpg.add_button(label="Load Model", tag="btn_load", width=200, callback=load_model_callback)
                dpg.add_text("No model yet. A trained model is kept and loaded again at the next start.",
                             tag="model_classes_text", color=C["muted"], wrap=0)


def build_demo_page():
    with dpg.child_window(tag="page_DEMO", border=False, show=False):
        with dpg.group(horizontal=True):
            build_product_panel()
            build_trial_panel()


def build_product_panel():
    with dpg.child_window(tag="product_panel", width=440, height=-1):
        dpg.add_text(PRODUCT["name"].upper(), tag="product_title")
        font("product_title", "h2")
        dpg.add_text(PRODUCT["tagline"], color=C["muted"], wrap=0)
        dpg.add_text(PRODUCT[T.NEUTRAL]["request"], tag="request_text", color=C["muted"])
        font("request_text", "banner")
        with dpg.group(horizontal=True):
            for role, draw in ((T.I1, draw_bottle), (T.I2, draw_bowl)):
                key = ROLE_TAG[role]
                with dpg.child_window(tag=f"tile_{key}", width=200, height=TILE_H, no_scrollbar=True):
                    with dpg.drawlist(width=170, height=60, tag=f"icon_{key}"):
                        draw(f"icon_{key}", 170, 60, C[key])
                    dpg.add_text(PRODUCT[role]["how"], color=C["muted"])
                    dpg.add_text(PRODUCT[role]["request"], tag=f"tile_{key}_request", color=C[key])
                    font(f"tile_{key}_request", "h2")
                    with dpg.group(horizontal=True):
                        dpg.add_text("0", tag=f"count_{key}")
                        font(f"count_{key}", "h2")
                        with dpg.group():
                            dpg.add_spacer(height=4)
                            dpg.add_text(PRODUCT[role]["counter"], color=C["muted"])
                dpg.bind_item_theme(f"tile_{key}", f"th_tile_{key}")
        dpg.add_text(PRODUCT["hint"], color=C["muted"], wrap=0)
        with dpg.group(horizontal=True):
            dpg.add_text("RECENT REQUESTS", color=C["muted"])
            dpg.add_button(label="Reset", tag="reset_counts_btn", small=True, callback=reset_counts_callback)
        with dpg.table(tag="log_table", header_row=False, policy=dpg.mvTable_SizingStretchProp, borders_innerH=True):
            dpg.add_table_column(init_width_or_weight=0.8)
            dpg.add_table_column(init_width_or_weight=0.7)
            dpg.add_table_column(init_width_or_weight=1.3)
            for k in range(LOG_ROWS):
                with dpg.table_row():
                    dpg.add_text("", tag=f"log_t_{k}", color=C["muted"])
                    dpg.add_text("", tag=f"log_l_{k}")
                    dpg.add_text("", tag=f"log_a_{k}", color=C["muted"])
        dpg.add_spacer(height=4)
        dpg.add_group(tag="spec_slot_demo")


def build_trial_panel():
    with dpg.child_window(tag="trial_panel", width=-1, height=-1):
        with dpg.group(horizontal=True):
            dpg.add_text("BAKEOFF TRIAL", tag="trial_title")
            font("trial_title", "h2")
            dpg.add_text("", tag="participant_text")
            font("participant_text", "bold")
            dpg.add_text("", tag="trial_progress_text", color=C["muted"])
        with dpg.group(horizontal=True):
            dpg.add_text("Practice ", color=C["muted"])
            for t in T.PRACTICE_TYPES:
                dpg.add_button(label="", tag=f"ttype_{TYPE_KEY[t]}", width=130, callback=trial_type_callback,
                               user_data=t)
        with dpg.group(horizontal=True):
            dpg.add_text("Positive", color=C["muted"])
            for t in (T.POS_I1, T.POS_I2):
                dpg.add_button(label="", tag=f"ttype_{TYPE_KEY[t]}", width=130, callback=trial_type_callback,
                               user_data=t)
            dpg.add_spacer(width=12)
            dpg.add_text("Negative", color=C["muted"])
            for t in (T.NEG_I1, T.NEG_I2):
                dpg.add_button(label="", tag=f"ttype_{TYPE_KEY[t]}", width=130, callback=trial_type_callback,
                               user_data=t)
        dpg.add_button(label="", tag="trial_banner", width=-1, height=76, callback=banner_callback)
        font("trial_banner", "banner")
        dpg.bind_item_theme("trial_banner", "th_banner_wait")
        with dpg.group(horizontal=True):
            dpg.add_text("", tag="trial_hint", color=C["muted"])
            dpg.add_button(label="Start anyway", tag="override_btn", show=False, callback=override_callback)
        dpg.add_text("", tag="trial_say", wrap=0)
        font("trial_say", "bold")
        dpg.add_text("", tag="trial_result", color=C["muted"], wrap=0)
        with dpg.table(tag="tally_table", header_row=True, policy=dpg.mvTable_SizingStretchProp,
                       borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            dpg.add_table_column(label="", init_width_or_weight=1.7)
            dpg.add_table_column(label=f"Interaction#1 ({PRODUCT[T.I1]['name']})", init_width_or_weight=1.0)
            dpg.add_table_column(label=f"Interaction#2 ({PRODUCT[T.I2]['name']})", init_width_or_weight=1.0)
            dpg.add_table_column(label="Accuracy", init_width_or_weight=0.8)
            for rk, rlabel in (("pos", "Positive"), ("neg", "Negative")):
                with dpg.table_row(height=36):
                    dpg.add_text(f"Successful trials for {rlabel}")
                    for col in ("1", "2", "acc"):
                        dpg.add_text("0", tag=f"tally_{rk}_{col}")
                        font(f"tally_{rk}_{col}", "h2")
        with dpg.group(horizontal=True):
            dpg.add_text("Overall 0/20", tag="tally_overall", color=C["muted"])
            dpg.add_button(label="Void last trial", tag="void_btn", callback=void_callback)
            dpg.add_button(label="Copy numbers", tag="copy_btn", callback=copy_numbers_callback)
            dpg.add_spacer(width=12)
            dpg.add_input_text(tag="next_pid_input", default_value="P2", hint="next ID", width=110)
            dpg.add_button(label="Next participant", tag="next_participant_btn", callback=next_participant_callback)
            dpg.bind_item_theme("next_participant_btn", "th_primary")
        dpg.add_text("", tag="roles_text", color=C["muted"], wrap=0)
        dpg.add_text("", tag="trial_log_text", color=C["muted"], wrap=0)


def build_dialogs():
    with dpg.window(tag="confirm_next", modal=True, show=False, no_title_bar=True, no_resize=True, no_move=True,
                    width=560, height=190):
        dpg.add_text("", tag="confirm_next_text", wrap=530)
        dpg.add_spacer(height=8)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start next participant", tag="confirm_next_yes", width=260,
                           callback=confirm_next_callback)
            dpg.add_button(label="Cancel", width=120, callback=lambda: dpg.configure_item("confirm_next", show=False))
    dpg.bind_item_theme("confirm_next_yes", "th_primary")

    with dpg.window(tag="void_dialog", modal=True, show=False, no_title_bar=True, no_resize=True, no_move=True,
                    width=560, height=230):
        dpg.add_text("", tag="void_text", wrap=530)
        dpg.add_input_text(tag="void_reason", default_value="experimenter error", label="reason", width=380)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Void it", tag="void_yes", width=160, callback=void_confirmed_callback)
            dpg.add_button(label="Cancel", width=120, callback=lambda: dpg.configure_item("void_dialog", show=False))
    dpg.bind_item_theme("void_yes", "th_danger")

    # File dialogs
    with dpg.file_dialog(tag="load_model_dialog", show=False, callback=load_model_dialog_callback, width=700,
                         height=400, modal=True, default_path=".", label="Choose Model to Load"):
        dpg.add_file_extension(".model")
    with dpg.file_dialog(tag="save_model_dialog", show=False, callback=save_model_dialog_callback,
                         label="Choose Save Location", width=700, height=400, modal=True, default_filename="model",
                         file_count=1, default_path="."):
        dpg.add_file_extension(".model")


def build_app():
    """Everything up to (not including) the window: fonts, themes, widgets, saved data, trial session."""
    global session
    dpg.create_context()
    load_fonts()
    build_themes()
    build_ui()
    dpg.set_primary_window("root", True)

    # loading initial dataclasses, the frames recorded for them last time, and the last calibration
    for c in metadata["data_classes"]:
        add_class(c)
    show_status(" ".join(m for m in (load_dataset(), load_calibration()) if m))

    session = T.TrialSession(role_map, log=T.TrialLog.for_session(RESULTS_DIR, defer_until_trial=True),
                             settings=trial_settings())
    session.next_participant("P1", time.monotonic())
    set_mode("SETUP" if floor_db is None else "TRAIN")
    last_model = Path(metadata["last_model_path"]) if metadata["last_model_path"] else None
    if last_model is not None and last_model.is_file():
        load_model_file(last_model, " (the last model used)")  # switches to DEMO when it is live
    refresh_product()
    refresh_trial_panel()


def main():
    init_audio()
    build_app()
    dpg.create_viewport(title=f"{PRODUCT['name']} - vibration bakeoff", width=1280, height=800, min_width=1024,
                        min_height=640, clear_color=rgba(C["bg"]))
    # Button callbacks run in the main loop below, between frames, instead of on Dear PyGui's own thread,
    # so they never overlap the spectrum update.
    dpg.configure_app(manual_callback_management=True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.maximize_viewport()  # show_viewport(maximized=True) leaves it at 1280x800 on Windows
    space, esc, f11 = KeyEdge(dpg.mvKey_Spacebar), KeyEdge(dpg.mvKey_Escape), KeyEdge(dpg.mvKey_F11)
    # Update every frame from this loop. (Re-registering a frame callback each frame, as the starter did,
    # silently stops for good when a slow step makes it miss its frame: the spectrum froze after Stop Recording.)
    while dpg.is_dearpygui_running():
        dpg.run_callbacks(dpg.get_callback_queue())
        run_main_jobs()
        poll_layout()
        now = time.monotonic()
        space_pressed, space_released = space.poll()  # polled every frame, so a press is never seen twice
        esc_pressed, _ = esc.poll()
        f11_pressed, _ = f11.poll()
        if f11_pressed:
            dpg.toggle_viewport_fullscreen()
        if ui_mode == "DEMO":
            if space_pressed:
                trial_press(now, override=shift_down())
            if space_released:
                session.space_up(now)
            if esc_pressed and not keyboard_busy():
                handle_trial_event(session.abort("cancelled with Esc", now))
        frame_callback()
        ui_tick(now)
        dpg.render_dearpygui_frame()
    save_dataset()  # keep frames recorded since the last Stop if the window is closed mid-recording
    if session.tally is not None and session.tally.trials:
        session.write_summary(time.monotonic())
    dpg.destroy_context()


if __name__ == "__main__":
    main()
