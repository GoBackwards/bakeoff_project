"""Throwaway go/no-go test for the Kitchen Chime Remote bakeoff design.

I1 = spoon tap on a steel bowl, I2 = spoon tap on a glass jar, Neutral = anything
that does not make either object ring. This script records labelled takes from the
laptop microphone and checks whether the two objects can be told apart from
everything else, using the same 50 ms Hamming FFT frames as the bakeoff app.

Commands (run from the repo root):
    uv run tools/gonogo.py plan                  print the list of takes to record
    uv run tools/gonogo.py devices               list input devices
    uv run tools/gonogo.py record                record takes interactively
    uv run tools/gonogo.py peaks audition_bowl1  show an object's strongest ring peaks
    uv run tools/gonogo.py analyze               fit modes, check G1-G6, print GO/NO-GO
    uv run tools/gonogo.py selftest              check the analysis on synthetic audio

Takes are WAV files in gonogo_runs/<session>/. The take name prefix decides how a
take is judged (see `plan`). Delete a take's WAV file to drop it from the analysis.
"""

import argparse
import datetime
import json
import math
import os
import queue
import shutil
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d

# ——— Settings ———

RUNS_DIR = Path(os.environ.get("GONOGO_RUNS_DIR", Path(__file__).resolve().parent.parent / "gonogo_runs"))
OBJECTS = ("bowl", "jar")

WINDOW_MS = 50  # same FFT window as models.get_ear()
FEATURE_FPS = 44  # new feature frames per second measured in the app
MODE_BAND_HZ = (300.0, 10000.0)  # where object modes are searched
N_MODES = 3
PROMINENCE_DB = 10.0  # a mode is "on" when this far above the local median
MEDIAN_BINS = 15  # local median over +/-7 bins
MODE_TOL_BINS = 2  # modes may drift +/-2 bins (about 40 Hz)
COMB_SPACING_BINS = (4, 50)  # harmonic spacings checked (80-1000 Hz): voices, hums, buzzes
COMB_WITHIN_DB = 10.0  # comb neighbours within this many dB of the peak make it a harmonic
MODES_NEEDED = 2  # an object is ringing when 2 of its modes are on
MAX_GAP_FRAMES = 1  # one dropout frame does not break a run
SAME_OBJECT_SEP_HZ = 300.0  # modes of one object are at least this far apart
MIN_MODE_SEP_HZ = 80.0  # modes of the two objects must be this far apart
PROFILE_S = (0.15, 0.45)  # modes are ranked by prominence this long after a strike
N_CANDIDATES = 6  # candidate peaks per object before dropping shared and comb-rejected ones
SHARED_PEAK_HZ = 25.0  # a peak at the same frequency in both objects' strikes is the spoon
MIN_ON_FRACTION = 0.5  # a mode must register (mode_on) in this share of its ring frames
RESTRIKE_RISE_DB = 9.0  # a strike on a still-ringing object counts if its modes jump this much

ONSET_JUMP_DB = 10.0
ONSET_ABOVE_FLOOR_DB = 12.0
ONSET_MODE_RISE_DB = 6.0
ONSET_KEEP_WITHIN_DB = 25.0  # onsets this far below the strongest in a take are noise
ONSET_REFRACTORY_S = 0.8
ONSET_TONAL_DB = 15.0  # a new ring shows several bins this far above the local median
ONSET_TONAL_BINS = 4
STRIKES_PLANNED = {"soft": 6, "firm": 7}  # strikes per bowl_*/jar_* take, from `plan`
AUDITION_STRIKES = 3

WARMUP_S = 0.5  # the Windows driver delivers ~0.4 s of silence when a stream opens
NEG_MIN_S = 5.0
REQUIRED_NEUTRAL_S = {"babble": 30.0, "talk": 20.0, "quiet": 15.0, "fan": 30.0}
REQUIRED_NEG = ("knock", "spoon", "keys", "scratch", "paper", "hiss", "laugh", "whistle", "hum", "typing", "ringtone")
OPTIONAL_NEG = ("fist", "lid", "spoondrop", "coin", "clap", "drum", "zipper", "velcro")
SPACE_MIN_PRESSES = 5
ISO_MIN_KNOCKS = 3
STEADY_TONE_DB = 15.0  # warn when a mode frequency carries a steady tone in negative takes

# GO criteria (K, the planned smoother length, is about 200 ms)
G2_SOFT_MIN_MS = 400
G2_FIRM_MIN_MS = 600
G3_WRONG_MAX_MS = 100
G4_MAX_MS = 100
G4_WARN_MS = 60
G4_MAX_WARN_TAKES = 2
G5_MAX_MS = 100
G1_FLAT_DB = 4.0  # +/-2 dB
G1_MIN_ACTIVE_S = 9.0
G1_MAX_TAIL_S = 2.0  # sound gone for longer than this before Enter: re-record...
G1_FAIL_TAIL_S = 4.0  # ...and longer than this: treat as removed by mic processing
G1_CLAP_DIP_DB = 3.0
G1_TONE_HZ = 2000.0
G1_TONE_PROM_DB = 20.0
G1_NOISE_MIN_DBFS = -55.0
G1_HF_MIN_DB = -30.0  # keys: 9-15 kHz band vs 1-8 kHz band at the loudest frames
G6_GATE_DB = 6.0
G6_MIN_DB = 3.0

TAKE_PLAN = """\
BEFORE YOU START
  - Laptop plugged in, sleep set to Never, Teams/Zoom/Discord closed.
  - Mic processing off: Armoury Crate AI Noise Cancelation (microphone), Windows
    audio enhancements / Voice Clarity / Studio Effects, Realtek noise suppression.
  - Start every take only after "Recording - go" appears.

[babble] = play crowd babble from a phone about 1 m away, loud enough that you
           raise your voice to talk across 1 m.   [quiet] = no babble.
Strikes: 2.5-3 s apart (3 s if the object rings longer than 2 s), and wait 3 s
after the last strike before pressing Enter. Vary who strikes, where on the
object, and height. Delete a bad take's WAV file to drop it.

  PROCESSING CHECK (G1) [quiet]; re-record to replace, the newest take counts.
  Press Enter while the phone is still playing.
    proc_noise       12 s of white or pink noise from the phone at 30 cm
    proc_tone        12 s of a 2 kHz tone from the phone at 30 cm
    proc_clap        2 kHz tone playing for 12 s; clap once, about 5 s in
    proc_keys        10 s of short key shakes (with pauses) 30 cm from the laptop

  CHOOSING OBJECTS [quiet] (then run: peaks audition_<name>)
    audition_<name>  3 soft spoon strikes on one candidate object, one take per object

  NEUTRAL (G4), all required
    neutral_babble   60 s [babble] only      neutral_quiet  30 s [quiet] room
    neutral_talk     30 s two people talking at the laptop [babble]
    neutral_fan      60 s [babble] with the CPU loaded so the fan spins up

  POSITIVES (G2, G3, G6) [babble], spoon unless noted, exactly this many strikes
    bowl_soft  6 strikes    bowl_med  7 strikes    bowl_firm  7 strikes
    jar_soft   6 strikes    jar_med   7 strikes    jar_firm   7 strikes
    bowl_knuckle / jar_knuckle  5 knuckle knocks each (information only)

  SPACE BAR (G4) [quiet]
    space            20 presses of the space bar (add space_usb for a USB keyboard)

  ANYTHING-BUT ACTIONS (G4) [babble], 10-15 s each, nothing touches bowl or jar
    required: neg_knock neg_spoon neg_keys neg_scratch neg_paper neg_hiss
              neg_laugh neg_whistle neg_hum neg_typing neg_ringtone
    optional: neg_fist neg_lid neg_spoondrop neg_coin neg_clap neg_drum
              neg_zipper neg_velcro
    (neg_spoon: spoon taps on the table, mat and palm rest; neg_whistle and
    neg_hum: slide slowly up and down through each object's pitch)

  ISOLATION (G5) [quiet]
    iso_bowl / iso_jar   hard knuckle knocks on the table 5 cm from the padded object

  LOOSE TOUCHES (G6; these are expected to ring) [babble]
    touch_bowl / touch_jar   nail flicks, fingertip taps, lift and set down,
                             palm rested then lifted
"""


# ——— Audio file helpers ———


def round_up_to_even(f):
    return int(math.ceil(f / 2.0) * 2)


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit WAV, got {8 * width}-bit")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if n_channels > 1:
        x = x.reshape(-1, n_channels)[:, 0]
    return x, rate


def load_audio(path):
    """Read a take and drop leading driver silence (up to 1 s)."""
    x, rate = read_wav(path)
    alive = np.flatnonzero(np.abs(x) > 3 / 32768.0)
    lead = min(alive[0] if len(alive) else 0, int(1.0 * rate))
    return x[lead:], rate


def write_wav(path, x, rate):
    pcm = np.clip(np.round(np.asarray(x) * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(pcm.tobytes())


def write_json_atomic(path, data):
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ——— Spectral analysis ———


class Spectrogram:
    """50 ms Hamming frames at the app's feature rate, matching models.get_ear()."""

    def __init__(self, x, rate):
        self.rate = rate
        self.n_win = round_up_to_even(rate * WINDOW_MS / 1000)
        self.hop = max(1, int(round(rate / FEATURE_FPS)))
        self.hop_s = self.hop / rate
        if len(x) < self.n_win:
            x = np.pad(x, (0, self.n_win - len(x)))
        frames = np.lib.stride_tricks.sliding_window_view(x, self.n_win)[:: self.hop]
        self.rms_db = (10.0 * np.log10(np.mean(frames**2, axis=1) + 1e-12)).astype(np.float32)
        mag = np.abs(np.fft.rfft(frames * np.hamming(self.n_win), axis=1))[:, 1:]
        self.db = (20.0 * np.log10(mag + 1e-9)).astype(np.float32)
        self.prom = self.db - median_filter(self.db, size=(1, MEDIAN_BINS), mode="nearest")
        self.prom_pool = maximum_filter1d(self.prom, 2 * MODE_TOL_BINS + 1, axis=1, mode="nearest")
        self.prom_pool1 = maximum_filter1d(self.prom, 3, axis=1, mode="nearest")
        self.db_pool = maximum_filter1d(self.db, 2 * MODE_TOL_BINS + 1, axis=1, mode="nearest")
        self.freqs = np.arange(1, mag.shape[1] + 1) * rate / self.n_win
        self.n_frames = self.db.shape[0]
        self._on = {}

    def frames(self, seconds):
        return int(round(seconds / self.hop_s))

    def bin_of(self, freq):
        return int(np.clip(round(freq * self.n_win / self.rate) - 1, 0, len(self.freqs) - 1))

    def bins_of_hz(self, hz):
        return max(1, int(round(hz * self.n_win / self.rate)))

    def band(self, lo, hi):
        hi = min(hi, self.rate / 2.0)
        return (self.freqs >= lo) & (self.freqs <= hi)

    def band_db(self, lo, hi):
        p = 10.0 ** (self.db[:, self.band(lo, hi)].astype(np.float64) / 10.0)
        return 10.0 * np.log10(p.sum(axis=1) + 1e-30)

    def mode_on(self, freq):
        """Per frame: a prominent peak near freq that started clean and is still ringing.

        A peak can only switch on when it is not part of a harmonic comb. Voices,
        hums and buzzes are combs: equally spaced peaks on both sides of a harmonic
        (or two below / two above the top / bottom one). Once on, it stays on while it
        stays prominent at the same frequency and does not grow, as a decaying ring does,
        even if babble briefly puts comb-like peaks around it.
        """
        b = self.bin_of(freq)
        if b in self._on:
            return self._on[b]
        nb = self.prom.shape[1]
        lo, hi = max(0, b - MODE_TOL_BINS), min(nb, b + MODE_TOL_BINS + 1)
        p = lo + np.argmax(self.prom[:, lo:hi], axis=1)
        rows = np.arange(self.n_frames)
        peak = self.prom[rows, p]
        level = self.db[rows, p]
        thr = np.maximum(PROMINENCE_DB, peak - COMB_WITHIN_DB)

        def strong(offset):
            return self.prom_pool1[rows, np.clip(p + offset, 0, nb - 1)] >= thr

        comb = np.zeros(self.n_frames, dtype=bool)
        for s in range(COMB_SPACING_BINS[0], COMB_SPACING_BINS[1] + 1):
            below, above = strong(-s), strong(s)
            comb |= (below & above) | (below & strong(-2 * s)) | (above & strong(2 * s))
        prominent = peak >= PROMINENCE_DB
        start = prominent & ~comb
        on = np.zeros(self.n_frames, dtype=bool)
        for t in range(self.n_frames):
            if start[t]:
                on[t] = True
            elif t and on[t - 1] and prominent[t] and abs(int(p[t]) - int(p[t - 1])) <= 1 and level[t] <= level[t - 1] + 3:
                on[t] = True
        self._on[b] = on
        return on

    def mode_hits(self, mode_freqs):
        """Boolean per frame: enough of the object's modes are on."""
        need = min(MODES_NEEDED, len(mode_freqs))
        return np.sum([self.mode_on(f) for f in mode_freqs], axis=0) >= need

    def mode_level(self, mode_freqs):
        """Mean absolute level (dB) of the modes per frame."""
        return np.mean([self.db_pool[:, self.bin_of(f)] for f in mode_freqs], axis=0)

    def _pick(self, cands, score, refractory_s=ONSET_REFRACTORY_S, keep_within_db=ONSET_KEEP_WITHIN_DB):
        """Strongest candidate per refractory window; drop ones far below the strongest."""
        if not cands:
            return []
        ref = self.frames(refractory_s)
        picked = []
        for t in sorted(cands, key=lambda t: (-round(float(score[t]), 1), t)):
            if all(abs(t - p) >= ref for p in picked):
                picked.append(t)
        best = max(score[t] for t in picked)
        kept = [t for t in picked if score[t] >= best - keep_within_db]
        # Report each event at its first frame, not where its score peaked.
        is_cand = set(cands)
        starts = []
        for t in kept:
            while t - 1 in is_cand:
                t -= 1
            starts.append(t)
        return sorted(starts)

    def generic_onsets(self):
        """Event starts without knowing the object: broadband jumps or new narrow peaks."""
        e = self.band_db(*MODE_BAND_HZ)
        floor = np.percentile(e, 10)
        prom = self.prom[:, self.band(*MODE_BAND_HZ)]
        cands = []
        for t in range(3, self.n_frames):
            lo = max(0, t - 4)
            jump = e[t] - e[lo:t].min() >= ONSET_JUMP_DB and e[t] >= floor + ONSET_ABOVE_FLOOR_DB
            new_peaks = (prom[t] >= ONSET_TONAL_DB) & (prom[lo:t].max(axis=0) < PROMINENCE_DB)
            if jump or int(new_peaks.sum()) >= ONSET_TONAL_BINS:
                cands.append(t)
        return self._pick(cands, maximum_filter1d(e, 5, origin=-2, mode="nearest"))  # max of e[t:t+5]

    def strike_onsets(self, mode_freqs):
        """Strikes of a known object: a click or mode jump where the object's modes rise."""
        level = self.mode_level(mode_freqs)
        hits = self.mode_hits(mode_freqs)
        quiet = level[~hits] if (~hits).sum() >= 10 else level
        floor = np.percentile(quiet, 10)
        e = self.band_db(*MODE_BAND_HZ)
        peak_after = maximum_filter1d(level, 4, origin=-2, mode="nearest")  # max of level[t:t+4]
        cands = []
        for t in range(3, self.n_frames):
            lo = max(0, t - 4)
            before = level[lo:t].min()
            if peak_after[t] - before < ONSET_MODE_RISE_DB or peak_after[t] < floor + ONSET_ABOVE_FLOOR_DB:
                continue
            if e[t] - e[lo:t].min() >= ONSET_JUMP_DB or level[t] - before >= ONSET_MODE_RISE_DB:
                cands.append(t)
        return self._pick(cands, peak_after)

    def transients(self):
        """Short clicks and knocks (key presses, knuckle knocks): a fast rise above 2 kHz
        that falls 6 dB within about 90 ms. Speech rarely does both."""
        hi = self.band_db(2000, 12000)
        cands = [
            t for t in range(3, self.n_frames - 4) if hi[t] - hi[t - 3 : t].min() >= 8 and hi[t] - hi[t + 4] >= 6
        ]
        return self._pick(cands, hi, refractory_s=0.25, keep_within_db=40)

    def knocks(self):
        """Knuckle knocks on the table: a broadband jump of 10 dB or more (no decay test,
        so a knock that sets an object ringing still counts)."""
        e = self.band_db(300, 12000)
        floor = np.percentile(e, 10)
        cands = [
            t
            for t in range(3, self.n_frames)
            if e[t] - e[t - 3 : t].min() >= ONSET_JUMP_DB and e[t] >= floor + ONSET_ABOVE_FLOOR_DB
        ]
        return self._pick(cands, e, refractory_s=0.25, keep_within_db=40)

    def steady_floor(self):
        """Per-bin prominence present in nearly every frame (5th percentile): steady tones."""
        if not hasattr(self, "_floor"):
            self._floor = np.maximum(np.percentile(self.prom, 5, axis=0), 0.0)
        return self._floor


def find_runs(mask, max_gap=MAX_GAP_FRAMES):
    """[start, end) runs of True, bridging gaps of up to max_gap False frames."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((start, prev + 1))
            start = i
        prev = i
    runs.append((start, prev + 1))
    return runs


def run_ms(run, spec):
    return (run[1] - run[0]) * spec.hop_s * 1000.0


def longest_run_ms(mask, spec):
    return max((run_ms(r, spec) for r in find_runs(mask)), default=0.0)


def strike_rings(spec, mode_freqs, planned=None):
    """Per strike: ring length (ms), onset level and rise (dB), overlap and censoring flags.

    An onset while the object is still ringing (a run that started 150+ ms earlier)
    counts as a new strike only if it has the spoon's click or the modes jump by
    RESTRIKE_RISE_DB. If more strikes than planned are found, the `planned` with the
    biggest rise are kept.
    """
    hits = spec.mode_hits(mode_freqs)
    runs = find_runs(hits)
    level = spec.mode_level(mode_freqs)
    e = spec.band_db(*MODE_BAND_HZ)
    settled = spec.frames(0.15)
    strikes = []
    for t0 in spec.strike_onsets(mode_freqs):
        lo = max(0, t0 - 4)
        rise = float(level[t0 : t0 + 4].max() - level[lo:t0].min())
        click = bool(e[t0 : t0 + 2].max() - e[lo:t0].min() >= ONSET_JUMP_DB)
        overlap = any(r[0] <= t0 - settled and r[1] > t0 - 2 for r in runs)
        if overlap and not (click or rise >= RESTRIKE_RISE_DB):
            continue
        cands = [r for r in runs if r[0] <= t0 + 6 and r[1] > t0]
        best = max(cands, key=lambda r: r[1], default=None)
        ring = run_ms((max(best[0], t0 - 1), best[1]), spec) if best else 0.0
        strikes.append(
            {
                "frame": t0,
                "ring_ms": ring,
                "level_db": float(level[t0 : t0 + 4].max()),
                "rise_db": rise,
                "overlap": overlap,
                "censored": bool(best is not None and best[1] >= spec.n_frames - 2),
            }
        )
    extra = 0
    if planned and len(strikes) > planned:
        extra = len(strikes) - planned
        strikes = sorted(sorted(strikes, key=lambda s: -s["rise_db"])[:planned], key=lambda s: s["frame"])
    return strikes, extra


def judged_rings(strikes, need):
    """Rings that count for G2: a ring cut off by the end of the take counts only if already long enough."""
    return [s["ring_ms"] for s in strikes if not (s["censored"] and s["ring_ms"] < need)]


def ring_profile(specs, mode_freqs=None, base=None):
    """Per strike: mean prominence 150-450 ms after onset (lasting peaks only), minus steady room tones.

    A room tone is subtracted only where it is also steady in that strike take, so a fan
    line heard only in neutral_fan cannot hide a real mode at the same frequency.
    """
    rows = []
    for spec in specs:
        onsets = spec.strike_onsets(mode_freqs) if mode_freqs else spec.generic_onsets()
        b = np.minimum(np.interp(spec.freqs, base[0], base[1]), spec.steady_floor()) if base is not None else 0.0
        a, z = spec.frames(PROFILE_S[0]), spec.frames(PROFILE_S[1])
        for t0 in onsets:
            seg = spec.prom[t0 + a : t0 + z]
            if len(seg) >= (z - a) // 2:
                rows.append((spec, t0, seg.mean(axis=0) - b))
    return rows


def pick_peaks(rows, n):
    """Strongest consistent, lasting peaks across strikes, sorted by score."""
    if not rows:
        return []
    ref = rows[0][0]
    grid = ref.freqs  # strikes may come from takes at different rates
    profiles, consistent = [], []
    for spec, _t0, prof in rows:
        p = np.interp(grid, spec.freqs, prof)
        profiles.append(p)
        consistent.append(maximum_filter1d(p, 2 * MODE_TOL_BINS + 1, mode="nearest") >= PROMINENCE_DB)
    mean_prom = np.mean(profiles, axis=0)
    consistency = np.mean(consistent, axis=0)
    sep = ref.bins_of_hz(SAME_OBJECT_SEP_HZ)
    local_max = mean_prom >= maximum_filter1d(mean_prom, 2 * sep + 1, mode="nearest")
    ok = ref.band(*MODE_BAND_HZ) & local_max & (mean_prom >= PROMINENCE_DB)
    score = np.where(ok, mean_prom * consistency, 0.0)
    picked = []
    for b in np.argsort(score)[::-1]:
        if score[b] <= 0 or len(picked) >= n:
            break
        if all(abs(b - p) >= sep for p in picked):
            picked.append(b)
    return [
        {"freq": float(grid[b]), "prominence_db": float(mean_prom[b]), "consistency": float(consistency[b])}
        for b in picked
    ]


def fit_object(specs, base, n=N_CANDIDATES):
    """Candidate modes, best first. Two passes: generic onsets find candidates, then strikes
    confirmed by those candidates. Each candidate gets on_frac: the share of its ring
    frames in which mode_on() registers it (comb-rejected peaks score low)."""
    rows = ring_profile(specs, None, base)
    peaks = pick_peaks(rows, n)
    if len(peaks) >= MODES_NEEDED:
        rows2 = ring_profile(specs, [p["freq"] for p in peaks[:N_MODES]], base)
        refined = pick_peaks(rows2, n)
        if len(refined) >= MODES_NEEDED:
            peaks, rows = refined, rows2
    for p in peaks:
        fracs = []
        for spec, t0, _ in rows:
            seg = spec.mode_on(p["freq"])[t0 + spec.frames(PROFILE_S[0]) : t0 + spec.frames(PROFILE_S[1])]
            if len(seg):
                fracs.append(seg.mean())
        p["on_frac"] = float(np.mean(fracs)) if fracs else 0.0
    return peaks


def select_modes(cands):
    """Object modes from per-object candidates: drop peaks shared by both objects (the
    spoon's own ring) and peaks that rarely register, then keep the best N_MODES."""
    notes, modes = [], {}
    shared = set()
    if all(cands.get(o) for o in OBJECTS):
        for a in cands["bowl"]:
            for b in cands["jar"]:
                if abs(a["freq"] - b["freq"]) <= SHARED_PEAK_HZ:
                    shared.add(("bowl", a["freq"]))
                    shared.add(("jar", b["freq"]))
        for f in sorted({f for o, f in shared if o == "bowl"}):
            notes.append(f"  ignored {f:.0f} Hz: it rings in both objects' strikes (probably the spoon)")
    for obj in OBJECTS:
        keep = [
            p["freq"]
            for p in cands.get(obj, [])
            if (obj, p["freq"]) not in shared and p["on_frac"] >= MIN_ON_FRACTION
        ]
        modes[obj] = keep[:N_MODES]
    return modes, notes


def mode_ring_ms(spec, t0, freq):
    """How long a single peak stays prominent after a strike, from when it first appears."""
    on = spec.prom_pool[t0:, spec.bin_of(freq)] >= PROMINENCE_DB
    first = np.flatnonzero(on[:4])
    if len(first) == 0:
        return 0.0
    rest = on[first[0] :]
    n = int(np.argmin(rest)) if not rest.all() else len(rest)
    return n * spec.hop_s * 1000.0


# ——— Take bookkeeping ———


def session_dir(name):
    return RUNS_DIR / (name or datetime.date.today().isoformat())


def classify(take):
    parts = take.lower().split("_")
    head = parts[0]
    sub = parts[1] if len(parts) > 1 else ""
    if head == "proc" and sub in ("noise", "tone", "clap", "keys"):
        return ("proc", sub)
    if head in OBJECTS:
        if sub == "soft":
            return ("pos", head, "soft")
        if sub in ("med", "medium", "firm", "hard"):
            return ("pos", head, "firm")
        if sub == "knuckle":
            return ("pos", head, "knuckle")
        return ("unknown",)
    if head == "touch" and sub in OBJECTS:
        return ("touch", sub)
    if head == "neg" and sub:
        return ("neg", "neg", sub)
    if head in ("space", "neutral"):
        return ("neg", head, sub)
    if head == "iso" and sub in OBJECTS:
        return ("iso", sub)
    if head == "audition":
        return ("audition",)
    return ("unknown",)


def load_takes(folder):
    meta = read_json(folder / "takes.json")
    takes, skipped = [], []
    for path in sorted(folder.glob("*.wav")):
        try:
            x, rate = load_audio(path)
        except (ValueError, EOFError, OSError, wave.Error) as e:
            skipped.append(f"{path.name} ({e})")
            continue
        info = meta.get(path.stem)
        info = info if isinstance(info, dict) else {}
        recorded = info.get("recorded") or datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        takes.append(
            {
                "name": path.stem,
                "kind": classify(path.stem),
                "spec": Spectrogram(x, rate),
                "seconds": len(x) / rate,
                "exclusive": bool(info.get("exclusive", False)),
                "recorded": str(recorded),
                "clipped": int(np.sum(np.abs(x) >= 0.999)),
                "silent": bool(len(x) < rate * 0.1 or np.max(np.abs(x)) < 3 / 32768.0),
            }
        )
    return takes, skipped


def mode_name(t):
    return "exclusive" if t["exclusive"] else "shared"


def pos_counted(t):
    """A strike take counts unless it is silent, a knuckle take, or too short for its planned strikes."""
    kind = t["kind"]
    if kind[0] != "pos" or kind[2] == "knuckle" or t["silent"]:
        return False
    return t["seconds"] >= STRIKES_PLANNED[kind[2]] * 2.0


# ——— Recording ———


class NoAudio(RuntimeError):
    pass


def resolve_device(sd, device, exclusive):
    """Device index from an index or a name substring (prefers MME, or WASAPI with --exclusive)."""
    devices = sd.query_devices()
    want = "WASAPI" if exclusive else "MME"

    def api_of(i):
        return sd.query_hostapis(devices[i]["hostapi"])["name"]

    inputs = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if device is None:
        default = sd.default.device[0]
        if not exclusive:
            return default
        prefix = devices[default]["name"][:31]
        match = [i for i in inputs if want in api_of(i) and devices[i]["name"].startswith(prefix)]
    elif device.isdigit():
        idx = int(device)
        if idx not in inputs:
            sys.exit(f"Device {idx} is not an input device. Run `devices`; numbers change when you plug things in.")
        return idx
    else:
        match = [i for i in inputs if want in api_of(i) and device.lower() in devices[i]["name"].lower()]
    if not match:
        sys.exit(f"No {want} input device matches. Run `devices` to see the inputs.")
    return match[0]


def open_input(args):
    import sounddevice as sd

    device = resolve_device(sd, args.device, args.exclusive)
    info = sd.query_devices(device)
    api = sd.query_hostapis(info["hostapi"])["name"]
    if args.exclusive:
        if "WASAPI" not in api:
            sys.exit("--exclusive needs a Windows WASAPI device. Run `devices` to see them.")
        rate = int(info["default_samplerate"])
        channels = int(info["max_input_channels"])
        extra = sd.WasapiSettings(exclusive=True)
    else:
        rate = args.rate
        channels = 1
        extra = None
    return sd, device, info, api, rate, channels, extra


def record_take(sd, device, rate, channels, extra, seconds=None):
    """Record one take after dropping the driver warm-up. Returns (samples, wall seconds)."""
    chunks = queue.Queue()
    state = {"skip": int(WARMUP_S * rate), "got": 0}

    def callback(indata, _frames, _time, status):
        if status:
            print(f"  [audio status: {status}]")
        x = indata[:, 0]
        if state["skip"] > 0:
            drop = min(state["skip"], len(x))
            state["skip"] -= drop
            x = x[drop:]
        if len(x):
            state["got"] += len(x)
            chunks.put(x.copy())

    with sd.InputStream(
        device=device, samplerate=rate, channels=channels, dtype="float32", callback=callback, extra_settings=extra
    ):
        waited = time.monotonic()
        while state["skip"] > 0:
            if time.monotonic() - waited > 3.0:
                raise NoAudio("no audio arrived from the device within 3 s")
            time.sleep(0.02)
        t_start, got_start = time.monotonic(), state["got"]
        if seconds is None:
            input("  Recording - go. Press Enter to stop. ")
        else:
            print(f"  Recording - go ({seconds:g} s)...")
            end = t_start + seconds
            while time.monotonic() < end:
                time.sleep(0.05)
        wall = time.monotonic() - t_start
        got_stop = state["got"]
    parts = []
    while not chunks.empty():
        parts.append(chunks.get())
    x = np.concatenate(parts) if parts else np.zeros(0)
    return x[got_start:got_stop], wall


def check_take(x, rate, wall, seconds=None):
    """Reasons not to save a take, or None if it looks sane."""
    got = len(x) / rate
    if got < 0.5 or (len(x) and np.max(np.abs(x)) == 0):
        return (
            f"only {got:.1f} s of audio arrived (or it was all zeros). Re-record; if it repeats, restart the "
            "script, check Settings > Privacy > Microphone, and close other apps using the mic."
        )
    if abs(got - wall) > max(0.05 * wall, 0.25):
        return (
            f"the audio driver delivered {got:.1f} s of audio in {wall:.1f} s (duplicated or dropped packets), "
            "so this take is unreliable. If you used --exclusive, use normal (shared) mode instead."
        )
    if seconds is not None and got < 0.9 * seconds:
        return f"only {got:.1f} s of the {seconds:g} s requested arrived"
    return None


def save_take(folder, take, x, rate, info, api, exclusive):
    folder.mkdir(parents=True, exist_ok=True)
    name, n = take, 2
    while (folder / f"{name}.wav").exists():
        name, n = f"{take}_{n}", n + 1
    write_wav(folder / f"{name}.wav", x, rate)
    meta_path = folder / "takes.json"
    meta = read_json(meta_path)
    meta[name] = {
        "device": info["name"],
        "hostapi": api,
        "rate": rate,
        "exclusive": exclusive,
        "seconds": round(len(x) / rate, 2),
        "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        write_json_atomic(meta_path, meta)
    except OSError as e:
        print(f"  warning: could not update takes.json ({e})")
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clipped = int(np.sum(np.abs(x) >= 0.999))
    peak_db = 20 * math.log10(peak) if peak > 0 else -120.0
    warn = "  CLIPPING: lower the input level and re-record" if clipped else ("  (very quiet)" if peak_db < -45 else "")
    print(f"  saved {name}.wav  {len(x) / rate:.1f} s  peak {peak_db:.1f} dBFS  clipped {clipped}{warn}")
    if classify(name) == ("unknown",):
        print("  note: unknown prefix, analyze will skip this take (see `plan`)")
    return name


def open_error(e, device, info, api):
    return (
        f"Could not record from device {device} ({info['name']}, {api}): {e}\n"
        "- Close Teams/Zoom/Discord and browser tabs that use the mic.\n"
        "- Without --exclusive, use an MME entry from `devices`.\n"
        "- For --exclusive: Sound settings > More sound settings > Recording > Microphone Array > Properties >\n"
        "  Advanced: tick both 'Allow applications to take exclusive control' boxes."
    )


def cmd_record(args):
    sd, device, info, api, rate, channels, extra = open_input(args)
    folder = session_dir(args.session)
    mode = "WASAPI exclusive" if args.exclusive else "shared"
    print(f"Device {device}: {info['name']} [{api}], {rate} Hz, {mode}")
    print(f"Saving to {folder}")
    if "MME" not in api and not args.exclusive:
        print("  note: the app opens MME devices; record with an MME entry to match it.")

    def one_take(take, seconds=None):
        try:
            x, wall = record_take(sd, device, rate, channels, extra, seconds)
        except sd.PortAudioError as e:
            sys.exit(open_error(e, device, info, api))
        except NoAudio as e:
            print(f"  ERROR: {e}. Take NOT saved. Restart the script; if it repeats, check the mic privacy setting.")
            return
        problem = check_take(x, rate, wall, seconds)
        if problem:
            print(f"  ERROR: {problem} Take NOT saved.")
            return
        save_take(folder, take, x, rate, info, api, args.exclusive)

    if args.take:
        try:
            one_take(args.take, args.seconds)
        except KeyboardInterrupt:
            print("\n  take discarded")
        return
    print("Type a take name (see `plan`), or press Enter on an empty line to quit.")
    while True:
        try:
            take = input("\nTake name: ").strip()
            if not take:
                break
            input("  Press Enter to start recording. ")
            one_take(take)
        except KeyboardInterrupt:
            print("\n  take discarded (type an empty name to quit)")
        except EOFError:
            break


# ——— Analysis: G1 processing check ———


def verdict(ok):
    return {True: "PASS", False: "FAIL", None: "MISSING"}[ok]


def active_window(spec, level, end, within_db, max_gap_s):
    """Longest stretch where the signal is within `within_db` of its loud level.

    Returns (start, stop, tail_s): measuring starts once the level has settled near
    its steady value (skipping fade-ins but keeping ramp-downs), and tail_s is how
    long the signal stayed gone before Enter was pressed.
    """
    if end <= 0:
        return None
    top = np.percentile(level[:end], 90)
    runs = find_runs(level[:end] >= top - within_db, max_gap=spec.frames(max_gap_s))
    if not runs:
        return None
    r = max(runs, key=lambda r: r[1] - r[0])
    settled = np.flatnonzero(level[r[0] : r[1]] >= top - 6)
    start = r[0] + (int(settled[0]) if len(settled) else 0) + spec.frames(0.2)
    stop = r[1] - spec.frames(0.2)
    return start, stop, (end - r[1]) * spec.hop_s


def g1_take(t):
    """Processing check for one proc_* take. Returns (True/False/None, line)."""
    spec, sub = t["spec"], t["kind"][1]
    end = spec.n_frames - spec.frames(0.3)  # ignore the Enter key at the end
    label = f"{t['name']} ({mode_name(t)})"
    if t["silent"]:
        return None, f"  {label}: silent take, re-record"
    if sub == "keys":
        return g1_keys(t, label)
    if t["seconds"] < G1_MIN_ACTIVE_S:
        return None, f"  {label}: only {t['seconds']:.1f} s long, not counted (record 12 s)"
    if sub == "noise":
        if np.percentile(spec.rms_db, 90) < G1_NOISE_MIN_DBFS:
            return None, f"  {label}: too quiet to judge (turn the phone up and re-record)"
        level = spec.band_db(500, 8000)
    else:
        b = spec.bin_of(G1_TONE_HZ)
        near = np.arange(max(0, b - 1), b + 2)
        sustained = minimum_filter1d(spec.prom[:, near], 5, axis=0, mode="nearest").max(axis=0)
        tone_bin = near[int(np.argmax(sustained))]
        if sustained.max() < G1_TONE_PROM_DB:
            return False, f"  {label}: no steady 2 kHz tone found. If the phone was playing it, mic processing removed it  FAIL"
        level = spec.db[:, tone_bin].astype(np.float64)
    clap = sub == "clap"
    win = active_window(spec, level, end, within_db=25 if clap else 10, max_gap_s=1.0 if clap else 0.07)
    if win is None:
        return None, f"  {label}: no signal found, re-record"
    start, stop, tail_s = win
    if tail_s > G1_FAIL_TAIL_S:
        return False, (
            f"  {label}: the sound dropped away {tail_s:.1f} s before Enter was pressed. If the phone was still"
            " playing, mic processing removed it; otherwise re-record and press Enter while it plays  FAIL"
        )
    if tail_s > G1_MAX_TAIL_S:
        return None, f"  {label}: the sound stopped {tail_s:.1f} s before Enter; re-record and press Enter while it plays"
    active_s = (stop - start) * spec.hop_s
    if active_s < G1_MIN_ACTIVE_S:
        return None, f"  {label}: only {active_s:.1f} s of steady signal, record 12 s"
    if not clap:
        block = spec.frames(0.5)
        blocks = [level[i : i + block].mean() for i in range(start, stop - block + 1, block)]
        spread = max(blocks) - min(blocks)
        ok = bool(spread <= G1_FLAT_DB)
        if sub == "tone" and ok:
            steady = float(np.mean(spec.prom[start:stop, tone_bin] >= G1_TONE_PROM_DB))
            if steady < 0.8:
                return False, f"  {label}: the tone kept fading out ({100 * steady:.0f}% of frames present)  FAIL"
        return ok, f"  {label}: level spread {spread:.1f} dB over {active_s:.0f} s (want <= {G1_FLAT_DB:.0f})  {verdict(ok)}"
    # Clap: measure the dip after every big transient (4-10 kHz, clear of the tone) and keep the worst.
    hi = spec.band_db(4000, 10000)
    prev_min = np.r_[hi[:1], minimum_filter1d(hi, 4, origin=1, mode="nearest")[:-1]]  # min of hi[t-4:t]
    jump = hi - prev_min
    lo_t, hi_t = max(start + spec.frames(1.0), spec.frames(2.0)), stop - spec.frames(1.5)
    if hi_t <= lo_t:
        return None, f"  {label}: too short to find the clap, record 12 s"
    biggest = float(jump[lo_t:hi_t].max())
    if biggest < ONSET_JUMP_DB:
        return None, f"  {label}: clap not found, clap once about 5 s into the take"
    claps = spec._pick(
        [t for t in range(lo_t, hi_t) if jump[t] >= max(ONSET_JUMP_DB, biggest - 10)], jump, refractory_s=0.5, keep_within_db=10
    )
    dips = []
    for c in claps:
        pre = level[c - spec.frames(2.0) : c - spec.frames(0.2)]
        calm = np.median(hi[c - spec.frames(2.0) : c - spec.frames(0.2)]) + 3
        settle = c + 1
        while settle < c + 5 and hi[settle] > calm:
            settle += 1
        post = level[settle : c + spec.frames(1.5)]
        if len(pre) >= 3 and len(post) >= 3:
            dips.append(float(np.median(pre) - median_filter(post, 3, mode="nearest").min()))
    if not dips:
        return None, f"  {label}: clap too close to the start or end, re-record"
    dip = max(dips)
    ok = bool(dip <= G1_CLAP_DIP_DB)
    return ok, f"  {label}: tone dip after clap {dip:.1f} dB (want <= {G1_CLAP_DIP_DB:.0f})  {verdict(ok)}"


def g1_keys(t, label):
    """Keys jingled in a quiet room must show energy above 8 kHz (no 16 kHz voice path)."""
    spec = t["spec"]
    if spec.rate / 2 < 10000:
        return None, f"  {label}: sample rate too low to check the high band"
    if np.percentile(spec.rms_db, 99) - np.percentile(spec.rms_db, 5) < 15:
        return None, f"  {label}: no key sounds found; shake the keys 30 cm from the laptop in a quiet room and re-record"
    loud = np.argsort(spec.rms_db)[-5:]
    hf = spec.band_db(9000, min(15000, spec.rate / 2 - 500))[loud]
    lf = spec.band_db(1000, 8000)[loud]
    rel = float(np.median(hf - lf))
    ok = bool(rel >= G1_HF_MIN_DB)
    note = "" if ok else ": something is cutting off sound above 8 kHz"
    return ok, f"  {label}: keys 9-15 kHz vs 1-8 kHz {rel:.0f} dB (want >= {G1_HF_MIN_DB:.0f}){note}  {verdict(ok)}"


def check_g1(takes):
    """Newest take of each kind per mode; G1 passes if every check passes in some mode."""
    subs_all = ("noise", "tone", "clap", "keys")
    latest, superseded = {}, 0
    for t in takes:
        if t["kind"][0] != "proc":
            continue
        key = (mode_name(t), t["kind"][1])
        if key in latest:
            superseded += 1
            if t["recorded"] < latest[key]["recorded"]:
                continue
        latest[key] = t
    lines, per_mode = [], {}
    for (mode, sub), t in sorted(latest.items()):
        ok, line = g1_take(t)
        lines.append(line)
        per_mode.setdefault(mode, {})[sub] = ok
    if superseded:
        lines.append(f"  ({superseded} older proc take(s) ignored; the newest of each counts)")
    results = {}
    for mode, subs in per_mode.items():
        vals = [subs.get(s) for s in subs_all]
        results[mode] = True if all(v is True for v in vals) else (False if any(v is False for v in vals) else None)
        missing = [s for s in subs_all if s not in subs]
        if missing:
            lines.append(f"  {mode}: not recorded yet: " + ", ".join(f"proc_{s}" for s in missing))
    passing = [m for m, v in results.items() if v is True]
    if passing:
        return True, lines, passing
    if any(v is False for v in results.values()):
        return False, lines, []
    return None, lines or ["  no proc_* takes yet"], []


# ——— Analysis: everything else ———


def env_baseline(takes):
    """Steady room/fan tones: highest median prominence per frequency across neutral takes (not talk)."""
    meds = [
        (t["spec"].freqs, np.median(t["spec"].prom, axis=0))
        for t in takes
        if t["kind"][:2] == ("neg", "neutral") and t["kind"][2] != "talk" and not t["silent"]
    ]
    if not meds:
        return None
    grid = meds[0][0]
    vals = np.max([np.interp(grid, f, m) for f, m in meds], axis=0)
    return grid, np.maximum(vals, 0.0)


def cmd_analyze(args):
    folder = session_dir(args.session)
    if not folder.exists():
        sys.exit(f"No session folder {folder}")
    report = analyze(folder)
    print(report)
    try:
        (folder / "report.txt").write_text(report, encoding="utf-8")
    except OSError as e:
        print(f"warning: could not write report.txt ({e})")


def analyze(folder):
    takes, skipped = load_takes(folder)
    out = [f"Go/no-go report for {folder}", ""]
    if skipped:
        out.append("Skipped unreadable files: " + "; ".join(skipped))
    if not takes:
        return "\n".join(out + ["No takes recorded yet."])

    base = env_baseline(takes)
    pos_takes = {obj: [t for t in takes if t["kind"][:2] == ("pos", obj) and pos_counted(t)] for obj in OBJECTS}
    cands = {obj: fit_object([t["spec"] for t in pos_takes[obj]], base) for obj in OBJECTS}
    modes, mode_notes = select_modes(cands)
    try:
        write_json_atomic(folder / "modes.json", modes)
    except OSError as e:
        out.append(f"warning: could not write modes.json ({e})")
    out.append("Object modes (fitted from this session's bowl_*/jar_* takes, saved to modes.json):")
    for obj in OBJECTS:
        out.append(f"  {obj:<5} " + (", ".join(f"{f:.0f} Hz" for f in modes[obj]) or "none yet"))
    out += mode_notes
    if base is None:
        out.append("  (no neutral_* takes yet, so steady room tones are not filtered out of the fit)")
    have_modes = all(len(modes[o]) >= MODES_NEEDED for o in OBJECTS)
    no_ring = [
        o
        for o in OBJECTS
        if len(modes[o]) < MODES_NEEDED and {t["kind"][2] for t in pos_takes[o]} >= {"soft", "firm"}
    ]

    sep_ok = None
    if have_modes:
        bin_hz = takes[0]["spec"].rate / takes[0]["spec"].n_win
        gap = min(
            min(abs(a - b) for a in modes["bowl"][:2] for b in modes["jar"]),
            min(abs(a - b) for a in modes["jar"][:2] for b in modes["bowl"]),
        )
        sep_ok = bool(gap >= MIN_MODE_SEP_HZ - 0.5 * bin_hz)
        out.append(f"  closest bowl/jar peaks: {gap:.1f} Hz apart (want >= {MIN_MODE_SEP_HZ:.0f})  {verdict(sep_ok)}")
        for t in takes:
            if t["kind"][0] == "neg" and not t["silent"]:
                spec = t["spec"]
                for obj in OBJECTS:
                    for f in modes[obj]:
                        steady = float(np.median(spec.prom_pool[:, spec.bin_of(f)]))
                        if steady >= STEADY_TONE_DB:
                            out.append(f"  warning: steady tone near {obj} mode {f:.0f} Hz in {t['name']} (+{steady:.0f} dB)")

    # Per-take table
    out += ["", f"{'take':<22} {'sec':>5} {'strikes':>7} {'bowl run':>9} {'jar run':>8}  notes"]
    stats = {}
    for t in takes:
        spec, kind = t["spec"], t["kind"]
        row = {"bowl": 0.0, "jar": 0.0, "strikes": [], "extra": 0, "clicks": None}
        if have_modes and kind[0] not in ("proc", "audition", "unknown") and not t["silent"]:
            for obj in OBJECTS:
                row[obj] = longest_run_ms(spec.mode_hits(modes[obj]), spec)
            if kind[0] == "pos":
                row["strikes"], row["extra"] = strike_rings(spec, modes[kind[1]], STRIKES_PLANNED.get(kind[2]))
        if kind[:2] == ("neg", "space") and not t["silent"]:
            row["clicks"] = len(spec.transients())
        elif kind[0] == "iso" and not t["silent"]:
            row["clicks"] = len(spec.knocks())
        stats[t["name"]] = row
        notes = []
        if kind == ("unknown",):
            notes.append("skipped: unknown prefix")
        elif kind[0] == "proc":
            notes.append("see G1")
        elif kind[0] == "audition":
            notes.append("use `peaks`")
        if t["silent"]:
            notes.append("SILENT, not counted")
        elif kind[0] == "pos" and kind[2] != "knuckle" and not pos_counted(t):
            notes.append(f"too short for {STRIKES_PLANNED[kind[2]]} strikes, not counted")
        if t["clipped"]:
            notes.append(f"CLIPPED {t['clipped']} samples: lower the input level")
        if row["strikes"] and kind[2] != "knuckle":
            need = G2_SOFT_MIN_MS if kind[2] == "soft" else G2_FIRM_MIN_MS
            rings = judged_rings(row["strikes"], need)
            if rings:
                notes.append(f"weakest ring {min(rings):.0f} ms")
        if row["clicks"] is not None:
            notes.append(f"{row['clicks']} knocks/presses detected")
        n_str = str(len(row["strikes"])) if kind[0] == "pos" else ""
        out.append(
            f"{t['name']:<22} {t['seconds']:>5.1f} {n_str:>7} {row['bowl']:>7.0f}ms {row['jar']:>6.0f}ms  {'; '.join(notes)}"
        )

    out += ["", "Criteria:"]
    g1, g1_lines, g1_modes = check_g1(takes)
    out.append(f"G1 processing: {verdict(g1)}" + (f" (passes in: {', '.join(g1_modes)})" if g1 else ""))
    out += g1_lines
    results = {"G1": g1, "sep": sep_ok, "mismatch": False}
    flags = {
        "clipped_pos": any(t["clipped"] for t in takes if t["kind"][0] == "pos"),
        "no_ring": no_ring,
        "voice_band": [(o, f) for o in OBJECTS for f in modes[o] if f < 1000],
    }
    if g1:
        other = {mode_name(t) for t in takes if t["kind"][0] in ("pos", "neg", "iso", "touch")}
        if other - set(g1_modes):
            results["mismatch"] = True
            out.append(
                f"  G1 passed only in {', '.join(g1_modes)} mode, but other takes were recorded in "
                f"{', '.join(sorted(other - set(g1_modes)))} mode: re-record them in the mode that passed."
            )

    if not have_modes:
        if no_ring:
            out.append(f"G2 ring length: FAIL")
            for o in no_ring:
                out.append(f"  the {o} does not ring clearly enough to find {MODES_NEEDED} lasting peaks: audition another object")
            results.update({"G2": False, "G3": None, "G4": None, "G5": None})
        else:
            out.append("G2-G6: need bowl_soft, bowl_med/firm, jar_soft and jar_med/firm takes with detectable strikes first.")
            results.update({"G2": None, "G3": None, "G4": None, "G5": None})
        out += ["", decision(results, flags)]
        return "\n".join(out)

    # G2 ring length and G3 separation
    pos = [t for t in takes if t["kind"][0] == "pos" and t["kind"][2] != "knuckle"]
    g2_ok, g3_ok, g2_lines, g3_lines = True, True, [], []
    for t in pos:
        obj, strength = t["kind"][1], t["kind"][2]
        if not pos_counted(t):
            g2_lines.append(f"  {t['name']}: silent or too short, not counted")
            continue
        need, planned = (G2_SOFT_MIN_MS if strength == "soft" else G2_FIRM_MIN_MS), STRIKES_PLANNED[strength]
        strikes, extra = stats[t["name"]]["strikes"], stats[t["name"]]["extra"]
        rings = judged_rings(strikes, need)
        if not rings:
            g2_ok = False
            g2_lines.append(f"  {t['name']}: no strikes detected  FAIL")
        else:
            ok = bool(min(rings) >= need) and len(strikes) >= planned
            g2_ok = g2_ok and ok
            g2_lines.append(
                f"  {t['name']}: {len(strikes)} of {planned} strikes, weakest ring {min(rings):.0f} ms (want >= {need})  {verdict(ok)}"
            )
            if len(strikes) < planned:
                g2_lines.append(
                    "    missing strikes rang too weakly to detect, were under 0.8 s apart, or rang into the next one"
                    f" (if you struck fewer than {planned}, re-record)"
                )
            if extra:
                g2_lines.append(f"    ignored {extra} weaker onset(s) (noise or double taps)")
            if len(rings) < len(strikes):
                g2_lines.append("    the last ring was cut off by Enter; wait 3 s after the last strike")
            overl = sum(s["overlap"] for s in strikes)
            if overl and not ok:
                g2_lines.append(f"    {overl} strike(s) landed while the object was still ringing; leave 3 s between strikes")
        wrong = "jar" if obj == "bowl" else "bowl"
        w = stats[t["name"]][wrong]
        ok = bool(w < G3_WRONG_MAX_MS)
        g3_ok = g3_ok and ok
        if not ok:
            g3_lines.append(f"  {t['name']}: {wrong} pattern for {w:.0f} ms (want < {G3_WRONG_MAX_MS})  FAIL")
    for obj in OBJECTS:
        have = {t["kind"][2] for t in pos_takes[obj]}
        for strength, names in (("soft", f"{obj}_soft"), ("firm", f"{obj}_med or {obj}_firm")):
            if strength not in have:
                g2_lines.append(f"  not recorded yet: {names}")
                g2_ok = None if g2_ok else g2_ok
                g3_ok = None if g3_ok else g3_ok
    out.append(f"G2 ring length: {verdict(g2_ok)}")
    out += g2_lines
    out.append(f"G3 separation: {verdict(g3_ok)}")
    out += g3_lines

    # G4 negatives
    g4_ok, warn_takes, g4_lines = True, 0, []
    neutral_s, neg_kinds, space_ok = {}, set(), False
    for t in takes:
        if t["kind"][0] != "neg":
            continue
        name, group, sub = t["name"], t["kind"][1], t["kind"][2]
        if t["seconds"] < NEG_MIN_S or t["silent"]:
            g4_lines.append(f"  {name}: too short or silent, not counted (record {NEG_MIN_S:.0f}+ s)")
            continue
        if group == "space":
            if stats[name]["clicks"] < SPACE_MIN_PRESSES:
                g4_lines.append(f"  {name}: only {stats[name]['clicks']} key presses detected, not counted (record it [quiet])")
                continue
            space_ok = True
        elif group == "neutral":
            neutral_s[sub] = neutral_s.get(sub, 0.0) + t["seconds"]
        else:
            neg_kinds.add(sub)
        worst = max(stats[name]["bowl"], stats[name]["jar"])
        if worst > G4_MAX_MS:
            g4_ok = False
            g4_lines.append(f"  {name}: object pattern for {worst:.0f} ms (want <= {G4_MAX_MS})  FAIL")
        elif worst > G4_WARN_MS:
            warn_takes += 1
            g4_lines.append(f"  {name}: {worst:.0f} ms (allowed, but close)")
    if warn_takes > G4_MAX_WARN_TAKES:
        g4_ok = False
        g4_lines.append(f"  {warn_takes} takes above {G4_WARN_MS} ms (want <= {G4_MAX_WARN_TAKES})  FAIL")
    missing = [f"neutral_{k} ({v:.0f}+ s)" for k, v in REQUIRED_NEUTRAL_S.items() if neutral_s.get(k, 0.0) < v]
    missing += [f"neg_{k}" for k in REQUIRED_NEG if k not in neg_kinds]
    if not space_ok:
        missing.append("space")
    if missing:
        g4_lines.append("  not recorded yet: " + ", ".join(missing))
        g4_ok = None if g4_ok else g4_ok
    out.append(f"G4 negatives: {verdict(g4_ok)}")
    out += g4_lines

    # G5 isolation
    g5_ok, g5_lines, iso_seen = True, [], set()
    for t in takes:
        if t["kind"][0] != "iso":
            continue
        name = t["name"]
        worst = max(stats[name]["bowl"], stats[name]["jar"])
        # An object that rings counts as a failure even if the knocks themselves were not detected.
        if t["silent"] or (stats[name]["clicks"] < ISO_MIN_KNOCKS and worst <= G5_MAX_MS):
            g5_lines.append(
                f"  {name}: {stats[name]['clicks'] or 0} knocks detected (want {ISO_MIN_KNOCKS}+), not counted: knock harder"
            )
            continue
        iso_seen.add(t["kind"][1])
        ok = bool(worst <= G5_MAX_MS)
        g5_ok = g5_ok and ok
        g5_lines.append(f"  {name}: {worst:.0f} ms (want <= {G5_MAX_MS})  {verdict(ok)}")
    for obj in OBJECTS:
        if obj not in iso_seen:
            g5_lines.append(f"  not recorded yet: iso_{obj}")
            g5_ok = None if g5_ok else g5_ok
    out.append(f"G5 isolation: {verdict(g5_ok)}")
    out += g5_lines
    results.update({"G2": g2_ok, "G3": g3_ok, "G4": g4_ok, "G5": g5_ok})

    # G6 firmness gap
    out.append("G6 firmness gap (soft criterion):")
    for obj in OBJECTS:
        soft = [
            s["level_db"]
            for t in takes
            if t["kind"] == ("pos", obj, "soft") and not t["silent"]
            for s in stats[t["name"]]["strikes"]
            if s["ring_ms"] > 0
        ]
        touches = [t for t in takes if t["kind"] == ("touch", obj) and t["seconds"] >= 3 and not t["silent"]]
        if not soft or not touches:
            out.append(f"  {obj}: need {obj}_soft and a 3+ s touch_{obj} take")
            continue
        rang = []
        for t in touches:
            hits = t["spec"].mode_hits(modes[obj])
            if hits.any():
                rang.append(float(t["spec"].mode_level(modes[obj])[hits].max()))
        if not rang:
            out.append(f"  {obj}: loose touches never made it ring, so no firmness gate is needed")
            continue
        gap = min(soft) - max(rang)
        if gap >= G6_GATE_DB:
            advice = "enable the firmness gate"
        elif gap >= G6_MIN_DB:
            advice = "OK without a firmness gate; expect Negative rows near 0.80"
        else:
            advice = "a light touch rings as loud as a soft strike" + ("; swap the jar for a steel cup" if obj == "jar" else "")
        out.append(f"  {obj}: softest strike is {gap:+.1f} dB vs loudest touch ({advice})")

    out += ["", decision(results, flags)]
    return "\n".join(out)


def decision(r, flags):
    hard = ["G1", "G2", "G3", "G4", "G5"]
    if any(r.get(k) is False for k in hard) or r.get("sep") is False:
        lines = ["DECISION: NO-GO"]
        if flags.get("clipped_pos") and (r.get("G2") is False or r.get("G3") is False):
            lines.append("- Some strike takes clipped: lower the input level and re-record them before judging G2/G3.")
        if r.get("G1") is False:
            lines.append("- G1: mic processing is still on. Turn off Armoury Crate AI Noise Cancelation, Windows audio")
            lines.append("  enhancements / Voice Clarity / Studio Effects and Realtek noise suppression, then re-record")
            lines.append("  the proc_* takes (the newest take counts). If it still fails, ask Yang before going on.")
        if flags.get("no_ring"):
            lines.append(f"- The {' and '.join(flags['no_ring'])} does not ring clearly: audition another object.")
        if r.get("sep") is False or r.get("G3") is False:
            lines.append("- Peaks too close or objects confused: audition other jars, a steel cup or a smaller bowl.")
        if r.get("G2") is False and not flags.get("no_ring"):
            lines.append("- Rings too short or strikes missed: pick objects that ring longer (thin steel, pad only under")
            lines.append("  the base), strike on the dot, and leave 3 s between strikes.")
        if r.get("G4") is False:
            lines.append("- A negative action produced an object pattern: re-pick objects whose modes avoid that sound,")
            lines.append("  or set MODES_NEEDED = 3 at the top of this script and re-run analyze.")
            for obj, f in flags.get("voice_band", []):
                lines.append(f"  The {obj}'s {f:.0f} Hz mode sits in the voice band; an object whose modes are all above")
                lines.append("  1 kHz holds up better against talking and babble.")
        if r.get("G5") is False:
            lines.append("- Table knocks reach the objects: use a thicker pad or sponge under each object.")
        return "\n".join(lines)
    todo = [k for k in hard if r.get(k) is None] + (["bowl/jar separation"] if r.get("sep") is None else [])
    if r.get("mismatch"):
        todo.append("takes in the mode that passed G1")
    if todo:
        return "DECISION: INCOMPLETE - still need takes for " + ", ".join(todo)
    return "DECISION: GO - every hard criterion (G1-G5) passed. Check G6 to decide on the firmness gate."


# ——— Other commands ———


def cmd_plan(_args):
    print(TAKE_PLAN)


def cmd_devices(args):
    import sounddevice as sd

    default_in = sd.default.device[0]
    print(f"{'idx':>4}  {'host API':<16} {'ch':>3} {'rate':>7}  name")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        api = sd.query_hostapis(d["hostapi"])["name"]
        if not args.all and "MME" not in api and "WASAPI" not in api:
            continue
        mark = "  <- default" if i == default_in else ""
        use = "use this" if "MME" in api else ("--exclusive only" if "WASAPI" in api else "")
        print(f"{i:>4}  {api:<16} {d['max_input_channels']:>3} {int(d['default_samplerate']):>7}  {d['name']}{mark}  [{use}]")
    print("\nRecord with an MME entry (what the app uses). Device numbers change when you plug things in;")
    print("--device also accepts part of the name, e.g. --device \"Microphone Array\". Use --all for every entry.")


def cmd_peaks(args):
    folder = session_dir(args.session)
    specs = []
    for take in args.takes:
        path = folder / f"{take}.wav"
        if not path.exists():
            sys.exit(f"Not found: {path}")
        try:
            x, rate = load_audio(path)
        except (ValueError, EOFError, OSError, wave.Error) as e:
            sys.exit(f"Cannot read {path.name}: {e}")
        specs.append(Spectrogram(x, rate))
    peaks = fit_object(specs, None)
    usable = [p for p in peaks if p["on_frac"] >= MIN_ON_FRACTION]
    if len(usable) < MODES_NEEDED:
        print("No clear lasting ring found. Strike a little harder, record in a quiet room, or try another object.")
        return
    top = [p["freq"] for p in usable[:N_MODES]]
    print(f"Strongest lasting ring peaks in {', '.join(args.takes)} (* = used as a mode):")
    print(f"\n{'freq Hz':>9}  {'prominence':>10}  {'in % of strikes':>15}  {'registers':>9}  {'ring ms':>8}")
    onsets = [(s, t0) for s in specs for t0 in s.strike_onsets(top)]
    for p in peaks[:5]:
        rings = [mode_ring_ms(s, t0, p["freq"]) for s, t0 in onsets]
        ring = np.median(rings) if rings else 0.0
        mark = "*" if p["freq"] in top else " "
        print(
            f"{p['freq']:>8.0f}{mark}  {p['prominence_db']:>8.1f} dB  {100 * p['consistency']:>14.0f}%"
            f"  {100 * p['on_frac']:>8.0f}%  {ring:>8.0f}"
        )
    rings = [r for spec in specs for r in judged_rings(strike_rings(spec, top)[0], G2_SOFT_MIN_MS)]
    print(f"\n{len(rings)} strike(s) found with the top {len(top)} peaks (plan: {AUDITION_STRIKES} per take).")
    if rings:
        print(f"2-of-3 ring length: min {min(rings):.0f} ms, median {np.median(rings):.0f} ms")
    print(f"Want: soft strikes >= {G2_SOFT_MIN_MS} ms. Longer is better; pick the object that rings longest.")


# ——— Self-test on synthetic audio ———


def _synth_takes(rate=44100, bowl_modes=(1180.0, 2950.0, 5300.0), jar_modes=(870.0, 2380.0, 4460.0), jar_tau=0.45, seed=7):
    """Synthetic session: realistic strikes (loud click, spoon tink, drifting modes) over noise + voiced babble."""
    rng = np.random.default_rng(seed)

    def room(n, level=0.002):
        return level * rng.standard_normal(n)

    def noise_babble(n, level=0.02):
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / rate)
        hiss = spec.copy()
        spec[(f < 200) | (f > 4000)] = 0
        hiss[(f < 4000) | (f > 9000)] = 0  # room/phone noise floor above the babble band, -14 dB
        b, hb = np.fft.irfft(spec, n), np.fft.irfft(hiss, n)
        b /= np.std(b) + 1e-12
        hb /= np.std(hb) + 1e-12
        return level * (b * (1 + 0.5 * np.sin(2 * np.pi * 3.1 * np.arange(n) / rate)) + 0.2 * hb)

    def talker(n, level, lo=100, hi=230, top_hz=5000, glide=None, tilt=0.7, jitter=0.0):
        """Voiced sound: harmonics with 1/h**tilt roll-off; tilt 0.7 and no jitter is a harsh, clean hum."""
        t = np.arange(n) / rate
        if glide:
            f0 = glide[0] * (glide[1] / glide[0]) ** (0.5 - 0.5 * np.cos(2 * np.pi * t / (n / rate)))
        else:
            f0 = rng.uniform(lo, hi) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t + rng.uniform(0, 6)))
        if jitter:
            j = np.convolve(rng.standard_normal(n), np.ones(220) / 220, mode="same")
            f0 = f0 * (1 + jitter * j / (np.std(j) + 1e-12))
        phase = 2 * np.pi * np.cumsum(f0) / rate
        sig = sum(np.sin(h * phase) / h**tilt for h in range(1, int(top_hz / f0.min())))
        if not glide:
            env = (np.sin(2 * np.pi * rng.uniform(3, 5) * t + rng.uniform(0, 6)) > -0.2).astype(float)
            sig = sig * np.convolve(env, np.ones(441) / 441, mode="same")
        return level * sig / (np.std(sig) + 1e-12)

    # Crowd babble: 8 talkers with natural pitch jitter and speech-like roll-off.
    voiced_bank = sum(talker(int(40 * rate), 0.0085, top_hz=4000, tilt=1.5, jitter=0.01) for _ in range(8))

    def babble(n):
        start = rng.integers(0, len(voiced_bank) - n) if n < len(voiced_bank) else 0
        return noise_babble(n) + np.resize(voiced_bank[start:], n)

    def ring(n, at, modes, amp, tau):
        x = np.zeros(n)
        t = np.arange(n - at) / rate
        for k, f in enumerate(modes):
            drift = f + rng.uniform(-15, 15)
            x[at:] += amp / (k + 1) ** 0.5 * np.exp(-t / (tau * (1 - 0.15 * k))) * np.sin(2 * np.pi * drift * t)
        x[at:] += 0.4 * amp * np.exp(-t / 0.1) * np.sin(2 * np.pi * 6500 * t)  # the spoon's own tink, in every strike
        click_n = int(0.02 * rate)
        click = min(6 * amp, 0.3)
        x[at : at + click_n] += rng.standard_normal(click_n) * click * np.exp(-np.arange(click_n) / (0.003 * rate))
        return x

    def no_clip(x):
        peak = np.max(np.abs(x))
        return x * (0.9 / peak) if peak > 0.9 else x

    def strikes(modes, amp, tau, count, spacing=2.6, amps=None):
        n = int((spacing * count + 3.0) * rate)
        x = babble(n)
        for i in range(count):
            x += ring(n, int((1 + spacing * i) * rate), modes, amps[i] if amps else amp, tau)
        return no_clip(x)

    def clicks(count, secs, amp=0.3, lowpass=False, quiet=False):
        n = int(secs * rate)
        x = room(n) if quiet else babble(n)
        for i in range(count):
            at = int((0.5 + i * (secs - 1) / count) * rate)
            m = int(0.02 * rate)
            burst = rng.standard_normal(m) * amp * np.exp(-np.arange(m) / (0.006 * rate))
            if lowpass:
                burst = np.convolve(burst, np.ones(8) / 8, mode="same") * 2
            x[at : at + m] += burst
        return no_clip(x)

    def bursts(secs, lo, hi, amp=0.08, count=5, dur=0.6):
        n = int(secs * rate)
        x = babble(n)
        for i in range(count):
            at = int((0.4 + i * (secs - 1) / count) * rate)
            seg = rng.standard_normal(int(dur * rate))
            spec = np.fft.rfft(seg)
            f = np.fft.rfftfreq(len(seg), 1 / rate)
            spec[(f < lo) | (f > hi)] = 0
            seg = np.fft.irfft(spec, len(seg))
            x[at : at + len(seg)] += amp * seg / (np.std(seg) + 1e-12)
        return x

    def ringtone(secs):
        n = int(secs * rate)
        x = babble(n)
        for i, f in enumerate([700, 1500, 3300, 1500, 700, 3300] * 2):
            at = int((0.3 + 0.6 * i) * rate)
            t = np.arange(int(0.5 * rate)) / rate
            x[at : at + len(t)] += 0.05 * np.exp(-t / 0.15) * np.sin(2 * np.pi * f * t)
        return x

    n12 = int(12 * rate)
    t12 = np.arange(n12) / rate
    tone = 0.1 * np.sin(2 * np.pi * 2000 * t12) + room(n12)
    clap = tone.copy()
    at = int(5 * rate)
    clap[at : at + int(0.02 * rate)] += 0.25 * rng.standard_normal(int(0.02 * rate))
    sweep_t = np.arange(int(8 * rate)) / rate
    sweep = 0.05 * np.sin(2 * np.pi * np.cumsum(500 * (6000 / 500) ** (0.5 - 0.5 * np.cos(2 * np.pi * sweep_t / 8))) / rate)
    n10 = int(10 * rate)
    takes = {
        "proc_noise": 0.05 * rng.standard_normal(n12),
        "proc_tone": tone,
        "proc_clap": clap,
        "proc_keys": clicks(15, 10, amp=0.2, quiet=True),
        "bowl_soft": strikes(bowl_modes, 0.05, 0.6, 6),
        "bowl_med": strikes(bowl_modes, 0.15, 0.6, 7),
        "bowl_firm": strikes(bowl_modes, 0.4, 0.6, 7),
        "jar_soft": strikes(jar_modes, 0.05, jar_tau, 6),
        "jar_med": strikes(jar_modes, 0.15, jar_tau, 7),
        "jar_firm": strikes(jar_modes, 0.4, jar_tau, 7),
        "touch_bowl": strikes(bowl_modes, 0.006, 0.6, 4),
        "touch_jar": strikes(jar_modes, 0.006, jar_tau, 4),
        "neutral_babble": babble(int(32 * rate)),
        "neutral_talk": babble(int(22 * rate)) + talker(int(22 * rate), 0.05, tilt=1.0, jitter=0.01),
        "neutral_quiet": room(int(16 * rate)),
        "neutral_fan": babble(int(32 * rate)) + 0.02 * np.sin(2 * np.pi * 3500 * np.arange(int(32 * rate)) / rate),
        "space": clicks(20, 15, amp=0.5, quiet=True),
        "neg_keys": clicks(15, 8, amp=0.2),
        "neg_knock": clicks(6, 8, amp=0.5, lowpass=True),
        "neg_spoon": clicks(10, 8, amp=0.3),
        "neg_scratch": bursts(10, 1500, 9000),
        "neg_paper": bursts(8, 800, 12000, amp=0.1),
        "neg_hiss": bursts(8, 3000, 12000, amp=0.06),
        "neg_laugh": babble(n10) + talker(n10, 0.06, lo=250, hi=350),
        "neg_whistle": babble(len(sweep_t)) + sweep,
        "neg_hum": babble(n10) + talker(n10, 0.06, glide=(100, 400)),
        "neg_typing": clicks(25, 8, amp=0.1),
        "neg_ringtone": ringtone(8),
        "iso_bowl": clicks(6, 10, amp=0.5, lowpass=True, quiet=True),
        "iso_jar": clicks(6, 10, amp=0.5, lowpass=True, quiet=True),
    }
    helpers = {"strikes": strikes, "babble": babble, "rng": rng}
    return takes, helpers, rate


def _write_session(folder, takes, rate):
    folder.mkdir(parents=True, exist_ok=True)
    for name, x in takes.items():
        write_wav(folder / f"{name}.wav", x, rate)


def cmd_selftest(_args):
    failures, bowl_true, jar_true = [], (1180.0, 2950.0, 5300.0), (870.0, 2380.0, 4460.0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        takes, h, rate = _synth_takes()
        _write_session(tmp / "good", takes, rate)
        report = analyze(tmp / "good")
        modes = read_json(tmp / "good" / "modes.json")
        for obj, truth in (("bowl", bowl_true), ("jar", jar_true)):
            for f in truth:
                if not modes.get(obj) or min(abs(f - g) for g in modes[obj]) > 30:
                    failures.append(f"{obj} mode {f:.0f} Hz not found (got {modes.get(obj)})")
        if "DECISION: GO" not in report:
            failures.append("synthetic good session did not give GO")

        def variant(name, expect, mutate):
            folder = tmp / name
            shutil.copytree(tmp / "good", folder)
            mutate(folder)
            rep = analyze(folder)
            lines = rep.splitlines()
            for line_start, needle in expect:
                if not any(l.lstrip().startswith(line_start) and needle in l for l in lines):
                    failures.append(f"{name}: expected a '{line_start}...' line containing '{needle}'")
            return rep

        def replace_jar(**kw):
            def mutate(folder):
                alt, _, _ = _synth_takes(**kw)
                _write_session(folder, {k: v for k, v in alt.items() if k.startswith(("jar_", "touch_jar"))}, rate)

            return mutate

        def chord(folder):
            n = int(8 * rate)
            t = np.arange(int(rate)) / rate
            x = h["babble"](n)
            x[int(3 * rate) : int(4 * rate)] += 0.05 * (np.sin(2 * np.pi * 1180 * t) + np.sin(2 * np.pi * 2950 * t))
            write_wav(folder / "neg_chord.wav", x, rate)

        def duck(folder):
            x, _ = read_wav(folder / "proc_clap.wav")
            x[int(5.05 * rate) : int(5.45 * rate)] *= 10 ** (-10 / 20)
            write_wav(folder / "proc_clap.wav", x, rate)

        def missing_strike(folder):
            amps = [0.05, 0.05, 0.05, 0.002, 0.05, 0.05]  # the 4th strike barely touches the bowl
            write_wav(folder / "bowl_soft.wav", h["strikes"](bowl_true, 0.05, 0.9, 6, amps=amps), rate)

        def incomplete(folder):
            for name in ("bowl_soft", "iso_jar", "space", "neutral_talk", "neg_hum"):
                (folder / f"{name}.wav").unlink()

        def long_bowl_and_jingle(folder):
            # A bowl that rings ~5 s (re-strikes land on a ringing bowl) and keys shaken almost continuously.
            for name, amp, count in (("bowl_soft", 0.05, 6), ("bowl_med", 0.15, 7), ("bowl_firm", 0.4, 7)):
                write_wav(folder / f"{name}.wav", h["strikes"](bowl_true, amp, 1.6, count), rate)
            n = int(11 * rate)
            keys = 0.002 * h["rng"].standard_normal(n)
            for i in range(10):
                a = int((0.3 + i) * rate)
                seg = h["rng"].standard_normal(int(0.8 * rate)) * 0.05
                keys[a : a + len(seg)] += np.diff(seg, prepend=0)  # bright jingle, 80% of the time
            write_wav(folder / "proc_keys.wav", keys, rate)

        no_go = ("DECISION", "NO-GO")
        reps = {
            "MUST PASS: long-ringing bowl, continuous key jingle": variant(
                "longbowl", [("DECISION", "GO -")], long_bowl_and_jingle
            ),
            "short-ringing jar": variant("short", [("G2 ring length", "FAIL"), no_go], replace_jar(jar_tau=0.08)),
            "jar that does not ring": variant("noring", [("G2 ring length", "FAIL"), no_go], replace_jar(jar_tau=0.02)),
            "jar mode 35 Hz from a bowl mode": variant(
                "close", [("closest bowl/jar peaks", "FAIL"), no_go], replace_jar(jar_modes=(1215.0, 2380.0, 4460.0))
            ),
            "chord on two bowl modes": variant("chord", [("neg_chord", "FAIL"), no_go], chord),
            "tone ducked 10 dB after clap": variant("duck", [("G1 processing", "FAIL"), no_go], duck),
            "soft strike that does not ring": variant("missing", [("bowl_soft", "FAIL"), no_go], missing_strike),
            "incomplete session": variant("incomplete", [("DECISION", "INCOMPLETE")], incomplete),
        }
    print(report)
    print("\n--- variants (all but the first must NOT give GO) ---")
    for name, rep in reps.items():
        print(f"{name}: " + next(l for l in rep.splitlines() if l.startswith("DECISION")))
    print("\nSELFTEST " + ("PASSED" if not failures else "FAILED:\n  " + "\n  ".join(failures)))
    sys.exit(1 if failures else 0)


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="print the takes to record").set_defaults(func=cmd_plan)
    dev = sub.add_parser("devices", help="list input devices")
    dev.add_argument("--all", action="store_true", help="also list DirectSound and WDM-KS entries")
    dev.set_defaults(func=cmd_devices)
    rec = sub.add_parser("record", help="record takes")
    rec.add_argument("--session", help="session folder name (default: today's date)")
    rec.add_argument("--device", help="input device index or part of its name (default: Windows default input)")
    rec.add_argument("--rate", type=int, default=44100, help="sample rate in shared mode (default 44100)")
    rec.add_argument("--exclusive", action="store_true", help="WASAPI exclusive mode (bypasses Windows effects)")
    rec.add_argument("--take", help="record one take non-interactively")
    rec.add_argument("--seconds", type=float, default=12.0, help="length of a --take recording (default 12)")
    rec.set_defaults(func=cmd_record)
    pk = sub.add_parser("peaks", help="strongest ring peaks of one or more takes")
    pk.add_argument("takes", nargs="+")
    pk.add_argument("--session")
    pk.set_defaults(func=cmd_peaks)
    an = sub.add_parser("analyze", help="check G1-G6 and print GO/NO-GO")
    an.add_argument("--session")
    an.set_defaults(func=cmd_analyze)
    sub.add_parser("selftest", help="run the analysis on synthetic audio (about 2 minutes)").set_defaults(func=cmd_selftest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
