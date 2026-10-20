import json
import os
from pathlib import Path
import pickle
import tempfile
import threading
import time

import numpy as np
import pyaudio

import dearpygui.dearpygui as dpg

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

FFT_Y_AXIS_MAX = 500_000

ear = None  # StreamAnalyzer instance or None

device_list = []  # Available input device names
device_indices_by_name = {}  # {device_name: system audio device index}

collected_data = {}  # { class_name: [data]}
record_val = None  # Currently selected class
is_recording = False  # Whether training data is currently being collected

model = None  # Loaded/trained model
infer_classes = []

DATASET_PATH = Path("metadata/dataset.pkl")  # recorded frames, kept across restarts
CALIBRATION_PATH = Path("metadata/calibration.pkl")  # quiet-room noise floor, gate and frames
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

# ——— Initialization ———
# Populate device_list with input-capable devices
audio = pyaudio.PyAudio()
for idx in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(idx)
    if info.get("maxInputChannels", 0) > 0:
        device_list.append(info["name"])
        device_indices_by_name[info["name"]] = idx

# Determine default device
default_device = device_list[0] if device_list else None
laptop_mic = next((name for name in device_list if "MacBook" in name and "Microphone" in name), None)
if laptop_mic:
    default_device = laptop_mic
saved = metadata["default_device"]
if saved in device_list:
    default_device = saved

# Instantiate audio reader ('ear') if possible
if default_device:
    device_idx = device_indices_by_name[default_device]
    ear = get_ear(device=device_idx)
else:
    ear = None

# data management


def publish_model(new_model, new_classes):
    """Make a trained or loaded model live. Model, classes and smoother change in one step,
    so the frame loop never pairs a model with the wrong class list."""
    global model, infer_classes, inference
    new_smoother = LabelSmoother(neutral_class(new_classes), metadata["smooth_frames"], EXIT_FRAMES, HOLD_FRAMES)
    model, infer_classes = new_model, new_classes
    inference = (new_model, new_classes, new_smoother)
    dpg.configure_item("infer_classes_list", items=new_classes)
    dpg.set_value("pred_text", "listening...")  # replaced on the next audio frame
    dpg.set_value("raw_pred_text", "")


last_status = None


def show_status(text):
    """One-line message for the user under the gate controls (also printed to the console)."""
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

    # maybe need to shuffle later

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
            for attempt in range(5):  # Windows: antivirus or sync tools can hold the file for a moment
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.1)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


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
        dpg.set_value(f"cls_{cls}", f"{cls}: {len(frames)}")
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
    show_status(message)


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
        dpg.set_value("recording_status", f"Collecting data: {record_val}{note}")
        dpg.configure_item("recording_status", color=(80, 180, 100))
    elif record_val:
        dpg.configure_item("record_button", label="Record Data")
        dpg.set_value("recording_status", f"Ready: {record_val}")
        dpg.configure_item("recording_status", color=(180, 180, 180))
    else:
        dpg.configure_item("record_button", label="Record Data")
        dpg.set_value("recording_status", "Select a class first")
        dpg.configure_item("recording_status", color=(180, 180, 180))


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
    # Create UI group for the new class
    with dpg.group(parent="class_list", tag=f"grp_{cls}"):
        dpg.add_text(f"{cls}: 0", bullet=True, tag=f"cls_{cls}")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select", callback=lambda: select_class_callback(cls))
            dpg.add_button(label="Clear", callback=lambda: clear_class(cls))
            dpg.add_button(label="Delete", callback=lambda: delete_class(cls))
        dpg.add_separator()
    dpg.set_value("class_input", "")

    # updates the metadata
    metadata["data_classes"] = list(collected_data.keys())
    save_metadata()


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
    dpg.set_value(f"cls_{cls}", f"{cls}: 0")
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


# Mode switching
def switch_mode(selected):
    """Toggle between Train and Infer UI groups."""
    train_visible = selected == "Train"
    if not train_visible:
        set_recording(False)  # never keep recording training data while inferring
    dpg.configure_item("train_group", show=train_visible)
    dpg.configure_item("infer_group", show=not train_visible)
    dpg.set_value("mode", selected)


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
    dpg.configure_item("train_button", enabled=False)
    dpg.configure_item("training_status_text", show=True)

    def _background_training():
        try:
            X, y, class_names = format_collected_data(data)
            X = preprocess_data(X)
            model_type = dpg.get_value("model_combo")

            # train and set the global model
            new_model, new_classes = train_model(X, y, class_names, model_type)
            new_model.trained_with_gate_ = gated  # saved with the model; it then needs a gate to predict
            publish_model(new_model, new_classes)
            counts = ", ".join(f"{k} {len(v)}" for k, v in data.items())
            show_status(" ".join([f"Trained on {counts} frames (Neutral class: {neutral})."] + notes))
            switch_mode("Infer")
        except Exception as e:
            show_status(f"Training failed: {e}")
        finally:
            dpg.configure_item("train_button", enabled=True)
            dpg.configure_item("training_status_text", show=False)

    threading.Thread(target=_background_training, daemon=True).start()


def calibrate_callback():
    """Measure ~3 s of the quiet room (fan, hum) and set the loudness gate above it."""
    global calibrating
    set_recording(False)
    calibrating = []
    show_status("Calibrating: keep the room quiet for 3 seconds...")


def finish_calibration():
    global gate_db, floor_db, calibration_frames, calibrating
    frames, calibrating = calibrating, None
    floor_db, gate_db, spread = calibrate_noise(frames, last_freqs)
    calibration_frames = frames
    save_calibration()
    dpg.set_value("gate_input", gate_db)
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


def smooth_input_callback(s, a, u):
    frames = max(1, int(a))
    metadata["smooth_frames"] = frames
    save_metadata()
    if inference is not None:
        inference[2].enter_frames = frames


def margin_input_callback(s, a, u):
    metadata["min_margin"] = max(0.0, float(a))
    save_metadata()


def load_model_callback():
    dpg.configure_item("load_model_dialog", show=True)


def save_model_callback():
    dpg.configure_item("save_model_dialog", show=True)


# File dialog callbacks
def load_model_dialog_callback(s, a, u):
    path = a.get("file_path_name", None)
    print(path)

    def _background_loading():
        try:
            p = Path(path)
            new_model, new_classes = load_model(p)
            publish_model(new_model, new_classes)
            show_status(f"Loaded {p.name}.")
        except Exception as e:
            show_status(f"Loading failed ({e}). Continuing with the current model.")

    if path:
        threading.Thread(target=_background_loading, daemon=True).start()


def save_model_dialog_callback(s, a, u):
    path = a.get("file_path_name", None)
    snapshot = inference  # model and classes that belong together

    def _background_saving():
        p = Path(path)
        try:
            save_model(snapshot[0], snapshot[1], p)
            show_status(f"Saved {p.name}.")
        except Exception as e:
            show_status(f"Saving failed: {e}")

    if snapshot is None:
        show_status("Train or load a model before saving.")
        return
    if path:
        threading.Thread(target=_background_saving, daemon=True).start()


def show_level(level, amps, calibrated):
    """Live loudness readout; also collects frames while Calibrate Quiet runs."""
    if calibrating is not None:
        calibrating.append(np.array(amps, copy=True))
        if len(calibrating) >= CALIBRATION_FRAMES:
            finish_calibration()
    if calibrated:
        dpg.set_value("level_text", f"Ring level {level:.1f} dB   |   gate {gate_db:.1f} dB")
    else:
        dpg.set_value("level_text", f"Level {level:.1f} dB   |   gate not set: press Calibrate Quiet")


def show_prediction(amps, loud, calibrated):
    """Frames below the gate are Neutral; louder frames go to the model; the smoother hides flicker."""
    live_model, classes, live_smoother = inference  # one consistent snapshot
    if getattr(live_model, "n_features_in_", len(amps)) != len(amps):
        dpg.set_value("pred_text", "model does not match this microphone: retrain")
        dpg.set_value("raw_pred_text", "")
        return
    if getattr(live_model, "trained_with_gate_", False) and not calibrated:
        # this model never saw quiet frames, so without a gate it would guess on silence
        dpg.set_value("pred_text", "press Calibrate Quiet")
        dpg.set_value("raw_pred_text", "")
        return
    raw = live_smoother.neutral
    if loud:
        neutral_index = classes.index(raw) if raw in classes else None
        X = preprocess_data(np.array([amps]))
        y_pred = int(predict_confident(live_model, X, neutral_index, metadata["min_margin"])[0])
        raw = classes[y_pred] if 0 <= y_pred < len(classes) else live_smoother.neutral
    shown = live_smoother.update(raw)
    dpg.set_value("pred_text", shown)
    dpg.set_value("raw_pred_text", f"(this frame: {raw})")
    if shown in classes:
        dpg.configure_item("infer_classes_list", default_value=shown)


last_error = None


def frame_callback():
    """Runs once per rendered frame from the main loop at the bottom of this file."""
    global last_fft_count, last_error, last_freqs
    try:
        # get the live data
        freqs, amps = get_audio_features(ear)

        # Plot FFT amplitude against its corresponding frequency in Hz.
        dpg.set_value("fft_series", [list(freqs), list(amps)])
        if len(freqs) and len(amps):
            peak_index = int(np.argmax(amps))
            peak_frequency = float(freqs[peak_index])
            peak_amplitude = float(amps[peak_index])
            dpg.set_value(
                "fft_status",
                f"Peak: {peak_frequency:.1f} Hz / amplitude {peak_amplitude:.3g}",
            )
            dpg.set_axis_limits("x_axis", 0, float(freqs[-1]))
            dpg.set_axis_limits("y_axis", 0, FFT_Y_AXIS_MAX)

        # The GUI redraws faster than new audio arrives; only use frames with a fresh FFT.
        fft_count = ear.num_ffts if ear is not None else dpg.get_frame_count()
        new_frame = fft_count != last_fft_count
        last_fft_count = fft_count

        if new_frame and len(amps):
            last_freqs = freqs
            # A calibration made with other microphone settings is paused, not deleted.
            calibrated = floor_db is not None and len(floor_db) == len(amps)
            if floor_db is not None and not calibrated:
                show_status("The calibration doesn't match these microphone settings: press Calibrate Quiet.")
            level = frame_level_db(amps, freqs, floor_db if calibrated else None)
            show_level(level, amps, calibrated)
            loud = not calibrated or level >= gate_db

            # recording data to collected data: once the gate is set, tap classes skip quiet frames
            # (Neutral keeps them, so the model learns that room noise is Neutral)
            if is_recording and record_val and (loud or record_val == neutral_class(list(collected_data))):
                collected_data[record_val].append(np.array(amps, copy=True))
                dpg.set_value(
                    f"cls_{record_val}", f"{record_val}: {len(collected_data[record_val])}"
                )

            # live prediction
            if (dpg.get_value("mode") == "Infer") and inference is not None:
                show_prediction(amps, loud, calibrated)
        last_error = None
    except Exception as e:
        if str(e) != last_error:  # report each new problem once instead of every frame
            last_error = str(e)
            show_status(f"Error: {e}")


# Resize handler
def on_vp_resize(s, a):
    W, H = dpg.get_viewport_width(), dpg.get_viewport_height()
    w2 = int(W * .95 * 0.7)
    h2 = .95 * H
    # Position and size windows dynamically
    dpg.configure_item("fft_vis", pos=(0, 0), width=w2, height=h2)
    dpg.configure_item("ctrl_win", pos=(w2, 0), width=W - w2, height=h2)
    for tag in ("recording_status", "status_text", "level_text"):
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, wrap=max(140, W - w2 - 24))


# ——— UI Construction ———

dpg.create_context()
dpg.create_viewport(title="FFT Trainer", width=800, height=600)

# Menu bar for device selection
with dpg.viewport_menu_bar():
    with dpg.menu(label="Options"):
        dpg.add_combo(
            items=device_list,
            label="Device",
            tag="device_combo",
            default_value=default_device,
            callback=device_select_callback,
            width=300,
        )  # TODO: ADD A CALLBACK

# FFT Visualization window
with dpg.window(
    tag="fft_vis", label="FFT Spectrum", no_move=True, no_resize=True, no_collapse=True
):
    dpg.add_text("Real-time FFT Signal", bullet=True)
    dpg.add_text("Waiting for microphone data...", tag="fft_status")
    with dpg.plot(label="FFT Plot", height=-1, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis", label="Frequency (Hz)")
        dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis", label="Amplitude")
        dpg.set_axis_limits("y_axis", 0, FFT_Y_AXIS_MAX)
        dpg.add_line_series([], [], label="FFT", parent="y_axis", tag="fft_series")

# Control window with Train/Infer groups
with dpg.window(
    tag="ctrl_win", label="Controls", no_move=True, no_resize=True, no_collapse=True
):
    # Mode switch combo
    dpg.add_text("Mode:")
    dpg.add_combo(
        items=["Train", "Infer"],
        default_value="Train",
        tag="mode",
        callback=lambda s, a, u: switch_mode(a),
    )
    dpg.add_separator()

    # Loudness gate: quieter frames are shown as Neutral and are not recorded for the tap classes.
    dpg.add_text("Waiting for audio...", tag="level_text", wrap=220)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Calibrate Quiet", callback=calibrate_callback)
        dpg.add_input_float(
            tag="gate_input",
            label="gate dB (Enter)",
            default_value=0.0,
            step=1.0,
            format="%.1f",
            width=100,
            on_enter=True,
            callback=gate_input_callback,
        )
    dpg.add_text(
        "Press Calibrate Quiet first (3 s of quiet).",
        tag="status_text",
        wrap=220,
        color=(230, 180, 80),
    )
    dpg.add_separator()

    # Training UI group
    with dpg.group(tag="train_group"):
        dpg.add_text("Model:")
        dpg.add_combo(
            items=MODEL_OPTIONS,
            default_value=metadata["model_type"],
            tag="model_combo",
            callback=model_select_callback,
            width=-1,
        )
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="class_input")
            dpg.add_button(label="Add Class", callback=add_class_callback)
        dpg.add_separator()
        dpg.add_text("Classes & Data:", bullet=True)
        dpg.add_child_window(tag="class_list", width=-1, height=200)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Clear All", callback=clear_all_classes)
            dpg.add_button(label="Delete All", callback=delete_all_classes)
        dpg.add_separator()
        dpg.add_text(f"Selected: {record_val}", tag="selected_class")
        dpg.add_button(
            label="Record Data",
            tag="record_button",
            enabled=False,
            callback=record_button_callback,
        )
        dpg.add_text(
            "Select a class first",
            tag="recording_status",
            color=(180, 180, 180),
            wrap=220,
        )
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Train Model", tag="train_button", callback=train_model_callback
            )
            dpg.add_text("Training...", tag="training_status_text", show=False)

    # Inference UI group
    with dpg.group(tag="infer_group", show=False):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Save Model", callback=save_model_callback, tag="btn_save"
            )
            dpg.add_button(
                label="Load Model", callback=load_model_callback, tag="btn_load"
            )
        dpg.add_text("Classes:", bullet=True)
        dpg.add_listbox(
            items=infer_classes,
            tag="infer_classes_list",
            num_items=5,
            default_value="sss",
        )
        with dpg.group(horizontal=True):
            dpg.add_text("Prediction: ")
            dpg.add_text("---", tag="pred_text")
        dpg.add_text("", tag="raw_pred_text", color=(150, 150, 150))
        dpg.add_input_int(
            tag="smooth_input",
            label="frames to switch",
            default_value=metadata["smooth_frames"],
            min_value=1,
            max_value=10,
            min_clamped=True,
            max_clamped=True,
            width=100,
            callback=smooth_input_callback,
        )
        dpg.add_input_float(
            tag="margin_input",
            label="sureness (Enter)",
            default_value=metadata["min_margin"],
            step=0.25,
            min_value=0.0,
            min_clamped=True,
            format="%.2f",
            width=100,
            on_enter=True,
            callback=margin_input_callback,
        )

# File dialogs
with dpg.file_dialog(
    tag="load_model_dialog",
    show=False,
    callback=load_model_dialog_callback,
    width=700,
    height=400,
    modal=True,
    default_path=".",
    label="Choose Model to Load",
):
    dpg.add_file_extension(".model")

with dpg.file_dialog(
    tag="save_model_dialog",
    show=False,
    callback=save_model_dialog_callback,
    label="Choose Save Location",
    width=700,
    height=400,
    modal=True,
    default_filename="model",
    file_count=1,
    default_path=".",
):
    dpg.add_file_extension(".model")

# Setup callbacks and start
dpg.set_viewport_resize_callback(on_vp_resize)
on_vp_resize(None, None)  # initial layout

# loading initial dataclasses, the frames recorded for them last time, and the last calibration
for c in metadata["data_classes"]:
    add_class(c)
show_status(" ".join(m for m in (load_dataset(), load_calibration()) if m))

# Button callbacks run in the main loop below, between frames, instead of on Dear PyGui's own thread,
# so they never overlap the spectrum update.
dpg.configure_app(manual_callback_management=True)
dpg.setup_dearpygui()
dpg.show_viewport()
# Update every frame from this loop. (Re-registering a frame callback each frame, as before, silently
# stops for good when a slow step makes it miss its frame: the spectrum froze after Stop Recording.)
while dpg.is_dearpygui_running():
    dpg.run_callbacks(dpg.get_callback_queue())
    frame_callback()
    dpg.render_dearpygui_frame()
save_dataset()  # keep frames recorded since the last Stop if the window is closed mid-recording
dpg.destroy_context()
