from typing import Any
from pathlib import Path

import json
import pickle
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from stream_analyzer import StreamAnalyzer

MODEL_LINEAR_SVC = "LinearSVC"
MODEL_POLY_SVC = "SVC poly degree 1"
MODEL_OPTIONS = [MODEL_LINEAR_SVC, MODEL_POLY_SVC]
DEFAULT_MODEL_TYPE = MODEL_LINEAR_SVC

# Saved with each model so a model trained on different preprocessing is not used silently.
PREPROCESS_VERSION = "log10-shape-v1"

# A class whose name contains one of these words is the Neutral class (case-insensitive).
NEUTRAL_WORDS = ("neutral", "nothing", "silence", "silent", "idle", "background", "noise", "ambient")

GATE_BAND_HZ = (300.0, 10000.0)  # where the struck objects ring
# Ring level = how far the 2 strongest bins in the band stand out (one ring partial is ~2 bins wide).
# Tuned on a simulation of the app's FFT pipeline: about 18 dB more sensitive to a one-mode bottle
# than a 10-bin average, with no false openings in quiet, a 12 dB louder room or a new fan tone.
LEVEL_TOP_BINS = 2
GATE_MARGIN_DB = 4.0  # gate sits this far above the typical quiet-room level


def spectrum_db(amps) -> np.ndarray:
    return 20.0 * np.log10(np.asarray(amps, dtype=np.float64) + 1.0)


def frame_level_db(amps, freqs=None, floor_db=None) -> float:
    """Loudness of one FFT frame in dB, used by the gate (quieter frames are Neutral).

    With a calibrated noise floor (per-bin quiet-room spectrum), it is how far the
    strongest bins between 300 Hz and 10 kHz stand out above the floor, compared with
    the typical bin. A narrow ring stands out even when soft, while a room or fan that
    simply gets louder everywhere does not. Without a floor it is the frame's overall level.
    """
    if floor_db is None or len(floor_db) != len(amps):
        return float(10.0 * np.log10(np.mean(np.asarray(amps, dtype=np.float64) ** 2) + 1e-12))
    excess = spectrum_db(amps) - floor_db
    if freqs is not None and len(freqs) == len(excess):
        freqs = np.asarray(freqs)
        band = (freqs >= GATE_BAND_HZ[0]) & (freqs <= GATE_BAND_HZ[1])
        if band.any():
            excess = excess[band]
    k = min(LEVEL_TOP_BINS, len(excess))
    return float(np.mean(np.partition(excess, -k)[-k:]) - np.median(excess))


def calibrate_noise(frames, freqs) -> tuple[np.ndarray, float, float]:
    """From frames of a quiet room: (per-bin noise floor in dB, gate in dB, spread in dB).

    The gate uses the median quiet level, so one click or cough during calibration does not
    push it up; `spread` (p99 - median) shows how steady the room was.
    """
    frames = np.asarray(frames, dtype=np.float64)
    floor_db = np.median(spectrum_db(frames), axis=0)
    levels = np.array([frame_level_db(f, freqs, floor_db) for f in frames])
    gate = float(np.median(levels) + GATE_MARGIN_DB)
    spread = float(np.percentile(levels, 99) - np.median(levels))
    return floor_db, gate, spread


def neutral_class(classes: list[str]) -> str:
    """Name of the Neutral class among `classes`, or "Neutral" if there is none."""
    names = [str(name) for name in classes]
    for name in names:
        if name.strip().lower() == "neutral":
            return name
    for name in names:
        words = re.split(r"[^a-z]+", name.lower())
        if any(word in NEUTRAL_WORDS for word in words):
            return name
    return "Neutral"


class LabelSmoother:
    """Turns per-frame guesses into a steady label.

    A tap is shown once `enter_frames` frames in a row are not Neutral (the most common
    tap class among them is shown) and then held for at least `hold_frames`. It goes back
    to Neutral after `exit_frames` Neutral frames in a row. Switching straight to the
    other tap class needs `enter_frames` frames of it in a row.
    """

    def __init__(self, neutral: str, enter_frames: int = 3, exit_frames: int = 3, hold_frames: int = 18):
        self.neutral = neutral
        self.enter_frames = max(1, int(enter_frames))
        self.exit_frames = max(1, int(exit_frames))
        self.hold_frames = max(0, int(hold_frames))
        self.reset()

    def reset(self):
        self.current = self.neutral
        self.streak = []  # recent non-Neutral guesses while showing Neutral
        self.shown = 0  # frames the current tap label has been shown
        self.neutral_run = 0
        self.other, self.other_count = None, 0

    def update(self, raw: str) -> str:
        if self.current == self.neutral:
            if raw == self.neutral:
                self.streak = []
                return self.current
            self.streak.append(raw)
            if len(self.streak) >= self.enter_frames:
                # most common tap class; ties go to the most recent frame
                self.current = max(reversed(self.streak), key=self.streak.count)
                self.streak, self.shown, self.neutral_run = [], 0, 0
                self.other, self.other_count = None, 0
            return self.current

        self.shown += 1
        if raw == self.neutral:
            self.neutral_run += 1  # Neutral frames don't cancel a pending switch to the other tap class
            if self.neutral_run >= self.exit_frames and self.shown >= self.hold_frames:
                self.reset()
            return self.current
        self.neutral_run = 0
        if raw == self.current:
            self.other_count = 0
            return self.current
        if raw == self.other:
            self.other_count += 1
        else:
            self.other, self.other_count = raw, 1
        if self.other_count >= self.enter_frames:
            self.current, self.shown = raw, 0
            self.other, self.other_count = None, 0
        return self.current


def get_ear(device: int) -> StreamAnalyzer:
    """Gets a stream analyzer object to capture the FFT of inputted audio waves.
    Investigate configurations here for different preprocessing options.

    Args:
        device (int): device index for input audio device to use.

    Returns:
        StreamAnalyzer: Class for capture and FFT processing of audio signals.
    """
    return StreamAnalyzer(device=device, n_frequency_bins=512, FFT_window_size_ms=50)


def make_dummy_spectrum(
    n_bins: int = 512, peaks: list[int] = [50, 150, 300], rate: int = 1000
) -> tuple[list[float], list[float]]:
    """Generates a dummy FFT spectrum.

    Args:
        n_bins (int, optional): Number of frequency bins to generate. Defaults to 512.
        peaks (list[int], optional): Specific bin locations for a peak. Defaults to [50, 150, 300].
        rate (int, optional): Rate of sampling for bins. Defaults to 1000.

    Returns:
        tuple[list[float],list[float]]: _description_
    """
    freqs = np.linspace(0, rate / 2, n_bins)
    amps = np.zeros_like(freqs)
    for p in peaks:
        amps += np.exp(-0.5 * ((freqs - p) / 5) ** 2) * 5
    amps += 0.2 * np.random.rand(n_bins)
    return freqs, amps


def get_audio_features(ear: StreamAnalyzer) -> tuple[list[float], list[float]]:
    """Function called every frame which retrieves the audio features generated
    from a given StreamAnalyzer class. Intended for students to make preprocessing
    changes. The visualizer plots the full FFT as frequency in Hz on the x axis
    and FFT amplitude on the y axis.

    Args:
        ear (StreamAnalyzer): Class for capture and FFT processing of audio signals

    Returns:
        tuple[list[float], list[float]]: List of frequency bin values and corresponding amplitudes
    """
    if ear is None:
        return make_dummy_spectrum()
    else:
        freqs, amps, _frequency_bin_centres, _frequency_bin_energies = ear.get_audio_features()
        return freqs, amps


# model training
def preprocess_data(X: np.ndarray) -> np.ndarray:
    """Preprocessing before training and prediction. Data will not be visualized
    but will enable specific normalizations/preprocessing techniques for machine learning.

    Args:
        X (np.ndarray): (N,D) Array of data with N total trials for D bins in each trial.
        Collected from either the collected data in training or individual frames from prediction

    Returns:
        np.ndarray: (N, D) the preprocessed array in the same shape.
    """
    # Log magnitudes, minus each frame's mean: the classifier sees spectral shape, not loudness,
    # so a soft and a hard tap on the same object look alike.
    X = np.log10(np.asarray(X, dtype=np.float64) + 1.0)
    return X - X.mean(axis=1, keepdims=True)


def train_model(
    X: np.ndarray,
    y: list[int] | np.ndarray,
    class_names: list[str],
    model_type: str = DEFAULT_MODEL_TYPE,
) -> tuple[Any, list[str]]:
    """Interface to train a specific model. Students should choose a model
    and then train it using the data from preprocess data sent in. Models can consist of interfaces
    from sklearn, pytorch, or other modules (from scratch or libraries).

    Args:
        X (np.ndarray): (N, D) Preprocessed data for N trials with D bin.
        y (list[int] | np.ndarray): (N,) Array of labels for each trial of X. Should be in integer form.
        class_names (list[str]): Decoded y values for a string name for each class.

    Returns:
        tuple[Any, list[str]]: The model and the list of class names for decoding later.
    """
    if model_type == MODEL_LINEAR_SVC:
        model = LinearSVC(
            C=1.0,
            tol=0.001,
            random_state=1,
            max_iter=10_000,
            dual="auto",
            class_weight="balanced",  # Neutral usually has many more frames than each tap class
        )
    elif model_type == MODEL_POLY_SVC:
        # Port from Java Weka.
        model = SVC(
            C=1.0,  # the same complexity parameter
            kernel="poly",  # polynomial kernel
            degree=1,  # exponent E = 1.0
            coef0=0.0,  # kernel constant term C = 0
            tol=0.001,  # stopping tolerance L = 0.001
            shrinking=True,  # use shrinking heuristics (Weka's SMO always uses it)
            probability=False,  # disable probability estimates (-V -1 in Weka)
            break_ties=False,  # no direct Weka equivalent; default is fine
            random_state=1,  # seed for any randomized parts (-W 1)
            max_iter=-1,  # no limit on iterations (-V -1 / -N 0 implies full optimization)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # The scaler is part of the saved model, so training and inference scale features the same way.
    model = Pipeline([("scale", StandardScaler()), ("svm", model)])
    model.fit(X, y)
    return model, class_names


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Interface for using the model to predict some given data.

    Args:
        model (Any): Some model. Same model as one you trained in train_model
        X (np.ndarray): (N, D) N captures of D bins of data.

    Returns:
        np.ndarray: (N, ) Array of predicted labels for each trial. Array should have only integers.
    """

    # returns the index of some class name
    y_pred = model.predict(X)
    return y_pred


def predict_confident(model: Any, X: np.ndarray, neutral_index: int | None, min_margin: float) -> np.ndarray:
    """Like predict(), but a tap the model is unsure about becomes Neutral.

    "Unsure" means the winning class's score is within `min_margin` of the runner-up. On this
    project's recordings, a margin of 0.5 cut false taps on everyday sounds by about a third
    while keeping most tap frames.
    """
    labels = np.asarray(predict(model, X))
    if neutral_index is None or min_margin <= 0 or not hasattr(model, "decision_function"):
        return labels
    scores = np.asarray(model.decision_function(X))
    if scores.ndim == 1:  # two classes: distance from the boundary
        margin = np.abs(scores)
    else:
        top2 = np.sort(scores, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
    return np.where((labels != neutral_index) & (margin < min_margin), neutral_index, labels)


def _format_c_float(value: float) -> str:
    value = float(np.float32(value))
    if not np.isfinite(value):
        raise ValueError("Cannot export non-finite model value to C header")
    return f"{value:.9g}f"


def _format_c_float_rows(rows: np.ndarray, indent: str = "    ") -> str:
    formatted_rows = []
    for row in rows:
        values = [_format_c_float(value) for value in row]
        lines = []
        for i in range(0, len(values), 8):
            lines.append(f"{indent}    " + ", ".join(values[i : i + 8]))
        formatted_rows.append(f"{indent}{{\n" + ",\n".join(lines) + f"\n{indent}}}")
    return ",\n".join(formatted_rows)


def _class_names_for_export(model_classes: np.ndarray, classes: list[str]) -> list[str]:
    names = []
    for class_id in model_classes:
        try:
            class_index = int(class_id)
        except (TypeError, ValueError):
            names.append(str(class_id))
            continue

        if 0 <= class_index < len(classes):
            names.append(str(classes[class_index]))
        else:
            names.append(str(class_id))
    return names


def export_model_header(model: Any, classes: list[str], save_path: Path):
    """Export a trained linear scikit-learn model to an ESP32-friendly C++ header.

    The generated header includes float32 weights, biases, class labels, and
    prediction helpers. It supports LinearSVC-style models with coef_,
    intercept_, and classes_ attributes.
    """
    scaler = None
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scale")
        model = model.named_steps["svm"]
    if not all(hasattr(model, attr) for attr in ("coef_", "intercept_", "classes_")):
        raise ValueError("Header export currently supports LinearSVC-style models only")

    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    if scaler is not None:
        # Fold the scaler into the linear model: w.(x - mean)/scale + b = (w/scale).x + (b - w.mean/scale)
        intercept = intercept - (coef * scaler.mean_ / scaler.scale_).sum(axis=1)
        coef = coef / scaler.scale_
    weights = coef.astype(np.float32)
    biases = intercept.astype(np.float32)
    model_classes = np.asarray(model.classes_)

    if weights.ndim != 2:
        raise ValueError("Expected model.coef_ to be a 2D array")
    if biases.ndim != 1:
        raise ValueError("Expected model.intercept_ to be a 1D array")
    if len(biases) != len(weights):
        raise ValueError("Model weights and biases have incompatible shapes")

    is_binary_linear = len(model_classes) == 2 and len(weights) == 1
    is_multiclass_linear = len(model_classes) == len(weights)
    if not (is_binary_linear or is_multiclass_linear):
        raise ValueError("Unsupported linear model shape for header export")

    class_ids = [int(class_id) for class_id in model_classes]
    class_names = _class_names_for_export(model_classes, classes)

    num_score_rows, num_features = weights.shape
    class_names_literal = ", ".join(json.dumps(name) for name in class_names)
    class_ids_literal = ", ".join(str(class_id) for class_id in class_ids)
    biases_literal = ", ".join(_format_c_float(value) for value in biases)
    weights_literal = _format_c_float_rows(weights)

    if is_binary_linear:
        predict_index_body = """    float score = vibration_model_score_row(0, features);
    return score >= 0.0f ? 1 : 0;"""
    else:
        predict_index_body = """    int best_index = 0;
    float best_score = vibration_model_score_row(0, features);

    for (int class_index = 1; class_index < VIBRATION_MODEL_NUM_CLASSES; class_index++) {
        float score = vibration_model_score_row(class_index, features);
        if (score > best_score) {
            best_score = score;
            best_index = class_index;
        }
    }

    return best_index;"""

    header = f"""#pragma once

#ifndef PROGMEM
#define PROGMEM
#endif

// Generated from a scikit-learn linear model.
// The feature vector must match the Python FFT preprocessing exactly:
// take log10(amplitude + 1) of every bin, then subtract the mean of all bins.
// The Python app also treats frames below its loudness gate as Neutral; the gate is not exported.
#define VIBRATION_MODEL_NUM_CLASSES {len(class_names)}
#define VIBRATION_MODEL_NUM_FEATURES {num_features}
#define VIBRATION_MODEL_NUM_SCORE_ROWS {num_score_rows}

static const int VIBRATION_MODEL_CLASS_IDS[VIBRATION_MODEL_NUM_CLASSES] PROGMEM = {{
    {class_ids_literal}
}};

static const char *const VIBRATION_MODEL_CLASS_NAMES[VIBRATION_MODEL_NUM_CLASSES] = {{
    {class_names_literal}
}};

static const float VIBRATION_MODEL_BIASES[VIBRATION_MODEL_NUM_SCORE_ROWS] PROGMEM = {{
    {biases_literal}
}};

static const float VIBRATION_MODEL_WEIGHTS[VIBRATION_MODEL_NUM_SCORE_ROWS][VIBRATION_MODEL_NUM_FEATURES] PROGMEM = {{
{weights_literal}
}};

static inline float vibration_model_score_row(
    int row,
    const float features[VIBRATION_MODEL_NUM_FEATURES]
) {{
    float score = VIBRATION_MODEL_BIASES[row];

    for (int feature_index = 0; feature_index < VIBRATION_MODEL_NUM_FEATURES; feature_index++) {{
        score += VIBRATION_MODEL_WEIGHTS[row][feature_index] * features[feature_index];
    }}

    return score;
}}

static inline int vibration_model_predict_index(
    const float features[VIBRATION_MODEL_NUM_FEATURES]
) {{
{predict_index_body}
}}

static inline int vibration_model_predict_id(
    const float features[VIBRATION_MODEL_NUM_FEATURES]
) {{
    return VIBRATION_MODEL_CLASS_IDS[vibration_model_predict_index(features)];
}}

static inline const char *vibration_model_predict_name(
    const float features[VIBRATION_MODEL_NUM_FEATURES]
) {{
    return VIBRATION_MODEL_CLASS_NAMES[vibration_model_predict_index(features)];
}}
"""

    with open(save_path, "w") as f:
        f.write(header)


def save_model(model: Any, classes: list[str], save_path: Path):
    """Interface to save your generated model.

    Args:
        model (Any): Model created from train_model
        classes (list[str]): List of class names that are used to decode model prediction integers.
        save_path (Path): File path to save the model. Can only end in .model.
    """

    payload = {
        "model": model,
        "classes": classes,
        "model_type": type(model).__name__,
        "preprocess": PREPROCESS_VERSION,
    }

    with open(save_path, "wb") as f:
        pickle.dump(payload, f)


def load_model(file_path: Path) -> tuple[Any, list[str]]:
    """Interface to load your model.

    Args:
        file_path (Path): Path of the model file to load. Can only end in .model

    Returns:
        tuple[Any, list[str]]: Model loaded and the list of class names for decoding
    """

    with open(file_path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("preprocess") != PREPROCESS_VERSION:
        raise ValueError("This model was trained with different preprocessing; retrain it in this version.")
    return payload["model"], payload["classes"]
