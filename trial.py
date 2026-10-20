"""Bakeoff trial logic for the spoon-tap demo. Pure Python, standard library only.

The live app already shows ONE steady label per fresh audio frame (gate + model +
sureness + 5-frame smoother). This module turns that stream plus SPACE presses into
bakeoff trials exactly as the assignment slides define them, keeps the per-participant
spreadsheet tally and writes an append-only CSV log. It never imports numpy, Dear PyGui
or the repo, so it can be tested without a microphone.

Summary of the rules (the full spec ships with this file):

* Roles: the app's Neutral class name -> Neutral; the class chosen as Interaction#1
  (default: the class named "bottle") -> Interaction#1; the class chosen as
  Interaction#2 (default "bowl") -> Interaction#2. A label that is a model class with no
  role is a wrong label (the trial FAILS). A label that is not a model class at all
  (status text such as "press Calibrate Quiet", None, "") is an app error frame.
* Verdict (slides): Pos I1 passes iff the set is {I1} or {Neutral, I1}; Pos I2 likewise.
  Neg I1 passes iff the set is {Neutral}, {I2} or {Neutral, I2}; Neg I2 likewise.
  Everything else fails, including the empty set, any set holding both interactions and
  any set holding an unknown label.
* A trial is INVALID (never PASS) when the window had no fresh audio frame, an app error
  frame, an audio gap longer than max_gap_s, a settings change, or was aborted, UNLESS
  the set already holds a label the trial type forbids: then it is a definite FAIL,
  because more frames can only add labels, never remove one.
* State machine READY -> RECORDING -> RESULT, driven by SPACE. The trial type is locked
  at the start press. A start is refused until the displayed label has been Neutral for
  neutral_hold_s (0.3 s) - the slides define a trial as Neutral -> X -> Neutral - unless
  the experimenter uses the explicit override. No grace tail and no minimum duration:
  only frames that arrive between the two presses are recorded; the only filter on
  presses is the key debounce.
* Tally: 4 cells x 5 counted trials; a 6th start in a full cell is refused; COMPLETE after
  20; practice is never counted; Void works on the most recent trial only and needs a
  reason; accuracy per row = successes / 10, as in the projected spreadsheet.
"""
from __future__ import annotations

import csv
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- roles
NEUTRAL = "Neutral"
I1 = "Interaction#1"
I2 = "Interaction#2"
ROLES = (NEUTRAL, I1, I2)

# --------------------------------------------------------------------------- trial types
POS_I1 = "Positive I1"
POS_I2 = "Positive I2"
NEG_I1 = "Negative I1"
NEG_I2 = "Negative I2"
PRACTICE_NEUTRAL = "Practice Neutral"
PRACTICE_I1 = "Practice I1"
PRACTICE_I2 = "Practice I2"
REAL_TYPES = (POS_I1, POS_I2, NEG_I1, NEG_I2)  # spreadsheet cell order
PRACTICE_TYPES = (PRACTICE_NEUTRAL, PRACTICE_I1, PRACTICE_I2)
TRIAL_TYPES = PRACTICE_TYPES + REAL_TYPES
TRIALS_PER_CELL = 5
TRIALS_PER_PARTICIPANT = TRIALS_PER_CELL * len(REAL_TYPES)  # 20
ROW_DENOMINATOR = 2 * TRIALS_PER_CELL  # 10 trials per spreadsheet row

PASS, FAIL, INVALID, SET_ONLY = "PASS", "FAIL", "INVALID", "SET ONLY"

# Accepted sets, copied from the slides. Anything else fails.
PASS_SETS: dict[str, tuple[frozenset, ...]] = {
    POS_I1: (frozenset({I1}), frozenset({NEUTRAL, I1})),
    POS_I2: (frozenset({I2}), frozenset({NEUTRAL, I2})),
    NEG_I1: (frozenset({NEUTRAL}), frozenset({I2}), frozenset({NEUTRAL, I2})),
    NEG_I2: (frozenset({NEUTRAL}), frozenset({I1}), frozenset({NEUTRAL, I1})),
}
# Practice I1/I2 are judged like the positive trial (shown, never counted).
# Practice Neutral has no verdict: only the set is shown.
JUDGED_AS = {PRACTICE_I1: POS_I1, PRACTICE_I2: POS_I2, **{t: t for t in REAL_TYPES}}
# Labels that can appear in some passing set; any other label is a definite FAIL.
ALLOWED = {t: frozenset().union(*sets) for t, sets in PASS_SETS.items()}


def verdict(trial_type: str, labels: Iterable[str]) -> bool:
    """True = PASS, exactly the slide rules. Unknown labels and the empty set fail."""
    judged = JUDGED_AS.get(trial_type)
    if judged is None:
        raise ValueError(f"no verdict for trial type {trial_type!r}")
    return frozenset(labels) in PASS_SETS[judged]


def definitely_fails(trial_type: str, labels: Iterable[str]) -> bool:
    """The set holds a label that no passing set contains, so no extra frame can rescue it."""
    judged = JUDGED_AS.get(trial_type)
    if judged is None:
        raise ValueError(f"no verdict for trial type {trial_type!r}")
    return bool(frozenset(labels) - ALLOWED[judged])


def format_set(labels: Iterable[str]) -> str:
    """'{Neutral, Interaction#1}': roles in fixed order, then any other label sorted."""
    s = set(labels)
    return "{" + ", ".join([r for r in ROLES if r in s] + sorted(s - set(ROLES))) + "}"


def judge(trial_type: str, labels: Iterable[str], *, frames: int, error_frames: int = 0,
          max_gap_s: float = 0.0, gap_limit_s: float = 0.5,
          extra_invalid: Sequence[str] = ()) -> tuple[str, str]:
    """(verdict, reason) for one finished window. verdict is PASS, FAIL, INVALID or SET ONLY."""
    if trial_type not in TRIAL_TYPES:
        raise ValueError(f"unknown trial type {trial_type!r}")
    s = frozenset(labels)
    invalid = []
    if frames <= 0:
        invalid.append("no fresh audio frames in the window")
    if error_frames > 0:
        invalid.append(f"{error_frames} app error frame(s)")
    if frames > 0 and max_gap_s > gap_limit_s:
        invalid.append(f"audio gap {max_gap_s:.2f} s")
    invalid.extend(extra_invalid)
    if trial_type == PRACTICE_NEUTRAL:
        return (INVALID, "; ".join(invalid)) if invalid else (SET_ONLY, "practice: set only")
    judged = JUDGED_AS[trial_type]
    if definitely_fails(trial_type, s):
        reason = "wrong label: " + ", ".join(sorted(s - ALLOWED[judged]))
        if invalid:
            reason += " (also " + "; ".join(invalid) + ")"
        return FAIL, reason
    if invalid:
        return INVALID, "; ".join(invalid)
    if verdict(trial_type, s):
        return PASS, ""
    if not s:
        return FAIL, "empty set"
    target = I1 if judged in (POS_I1, NEG_I2) else I2
    return FAIL, f"{target} never shown"  # only a positive trial can get here


# --------------------------------------------------------------------------- role map
# Same rule as models.neutral_class, so the module agrees with the app's smoother.
NEUTRAL_WORDS = ("neutral", "nothing", "silence", "silent", "idle", "background", "noise", "ambient")
DEFAULT_I1_WORD = "bottle"
DEFAULT_I2_WORD = "bowl"
ROLE, UNMAPPED, ERROR = "role", "unmapped", "error"


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[^a-z]+", name.lower()) if w]


def detect_neutral(classes: Sequence[str]) -> str:
    """Name of the Neutral class, or 'Neutral' if there is none (mirrors models.neutral_class)."""
    names = [str(n) for n in classes]
    for n in names:
        if n.strip().lower() == "neutral":
            return n
    for n in names:
        if any(w in NEUTRAL_WORDS for w in _words(n)):
            return n
    return "Neutral"


def _find_class(classes: Sequence[str], word: str, exclude: set) -> str | None:
    """The one class named `word` (or with `word` as a whole word); None if absent or ambiguous."""
    exact = [c for c in classes if c.strip().lower() == word and c not in exclude]
    if len(exact) == 1:
        return exact[0]
    hits = [c for c in classes if word in _words(c) and c not in exclude]
    return hits[0] if len(hits) == 1 else None


@dataclass(frozen=True)
class RoleMap:
    classes: tuple
    neutral: str
    i1: str | None
    i2: str | None

    @classmethod
    def from_classes(cls, classes: Sequence[str], i1: str | None = None, i2: str | None = None,
                     neutral: str | None = None) -> "RoleMap":
        """Neutral by name; I1/I2 as chosen in the UI, else the 'bottle' / 'bowl' class.
        There is no order-based guess: an unresolved role blocks trials until it is chosen."""
        names = tuple(str(c) for c in classes)
        neutral = detect_neutral(names) if neutral is None else str(neutral)
        if i1 is None:
            i1 = _find_class(names, DEFAULT_I1_WORD, {neutral})
        if i2 is None:
            i2 = _find_class(names, DEFAULT_I2_WORD, {neutral})
        return cls(names, neutral, i1, i2)

    def problems(self) -> list[str]:
        """Blocking problems; trials cannot start until this is empty."""
        out = []
        for role, name in ((I1, self.i1), (I2, self.i2)):
            if not name:
                out.append(f"choose the class for {role}")
            elif name not in self.classes:
                out.append(f"{role} class {name!r} is not in the model")
            elif name == self.neutral:
                out.append(f"{role} class {name!r} is the Neutral class")
        if self.i1 and self.i1 == self.i2:
            out.append("Interaction#1 and Interaction#2 are the same class")
        return out

    def warnings(self) -> list[str]:
        out = []
        if self.neutral not in self.classes:
            out.append("the model has no Neutral class: Neutral comes only from the gate and sureness")
        extra = [c for c in self.classes if c not in (self.neutral, self.i1, self.i2)]
        if extra:
            out.append(f"classes with no role: {extra}; if one is shown during a trial, the trial FAILS")
        return out

    def role_of(self, label) -> tuple[str, str | None]:
        """('role', Neutral/I1/I2) | ('unmapped', 'Unmapped(name)') | ('error', None)."""
        if not isinstance(label, str) or not label:
            return ERROR, None
        if label == self.neutral:
            return ROLE, NEUTRAL
        if self.i1 is not None and label == self.i1:
            return ROLE, I1
        if self.i2 is not None and label == self.i2:
            return ROLE, I2
        if label in self.classes:
            return UNMAPPED, f"Unmapped({label})"
        return ERROR, None

    def as_dict(self) -> dict:
        return {"neutral": self.neutral, "i1": self.i1, "i2": self.i2}


# --------------------------------------------------------------------------- results / tally
@dataclass
class TrialResult:
    trial_no: int
    participant: str
    trial_type: str
    labels: frozenset
    verdict: str            # PASS / FAIL / INVALID / SET ONLY
    reason: str
    counted: bool
    t_start: float
    t_stop: float
    frames: int             # fresh audio frames in the window (error frames included)
    error_frames: int
    max_gap_s: float
    override_neutral: bool  # started with the override while the Neutral hold was not met
    quiet_before_start_s: float
    shown_seq: list         # displayed label per frame ('<...>' marks an app error frame)
    raw_seq: list           # optional per-frame model guess before smoothing
    settings: dict
    roles: dict
    timestamp: str
    voided: bool = False
    void_reason: str = ""

    @property
    def practice(self) -> bool:
        return self.trial_type in PRACTICE_TYPES

    @property
    def duration_s(self) -> float:
        return self.t_stop - self.t_start

    @property
    def set_text(self) -> str:
        return format_set(self.labels)

    @property
    def success(self) -> bool:
        return self.counted and not self.voided and self.verdict == PASS

    def display_text(self) -> str:
        head = f"#{self.trial_no} {self.trial_type}: {self.set_text}  {self.verdict}"
        tail = []
        if self.reason and self.verdict != SET_ONLY:
            tail.append(self.reason)
        if self.practice:
            tail.append("practice, not counted")
        elif not self.counted and not self.voided:
            tail.append("not counted")
        if self.voided:
            tail.append(f"VOIDED: {self.void_reason}")
        return head + (f"  ({'; '.join(tail)})" if tail else "")


class Tally:
    """One participant = one block of the projected spreadsheet."""

    def __init__(self, participant: str):
        self.participant = participant
        self.trials: list[TrialResult] = []

    def counted(self, cell: str) -> list[TrialResult]:
        return [t for t in self.trials if t.trial_type == cell and t.counted and not t.voided]

    def n_counted(self, cell: str) -> int:
        return len(self.counted(cell))

    def successes(self, cell: str) -> int:
        return sum(1 for t in self.counted(cell) if t.verdict == PASS)

    def full(self, cell: str) -> bool:
        return self.n_counted(cell) >= TRIALS_PER_CELL

    @property
    def n_counted_total(self) -> int:
        return sum(self.n_counted(c) for c in REAL_TYPES)

    @property
    def complete(self) -> bool:
        return all(self.full(c) for c in REAL_TYPES)

    @property
    def real_started(self) -> bool:
        return any(t.trial_type in REAL_TYPES and not t.voided for t in self.trials)

    def practice_done(self, practice_type: str) -> bool:
        return any(t.trial_type == practice_type and not t.voided and t.verdict != INVALID
                   for t in self.trials)

    def missing_practice(self) -> list[str]:
        return [p for p in PRACTICE_TYPES if not self.practice_done(p)]

    @property
    def invalid_count(self) -> int:
        return sum(1 for t in self.trials if t.verdict == INVALID and not t.voided)

    @property
    def voided_count(self) -> int:
        return sum(1 for t in self.trials if t.voided)

    def positive_accuracy(self) -> float:
        return (self.successes(POS_I1) + self.successes(POS_I2)) / ROW_DENOMINATOR

    def negative_accuracy(self) -> float:
        return (self.successes(NEG_I1) + self.successes(NEG_I2)) / ROW_DENOMINATOR

    def sheet_rows(self) -> list[list]:
        return [
            ["Successful trials for Positive", self.successes(POS_I1), self.successes(POS_I2),
             self.positive_accuracy()],
            ["Successful trials for Negative", self.successes(NEG_I1), self.successes(NEG_I2),
             self.negative_accuracy()],
        ]

    def sheet_text(self) -> str:
        lines = [f"Participant {self.participant}   (Interaction#1, Interaction#2, Accuracy)"]
        for label, a, b, acc in self.sheet_rows():
            lines.append(f"{label}: {a}, {b}, {acc:.1f}")
        lines.append(f"counted {self.n_counted_total}/{TRIALS_PER_PARTICIPANT}, "
                     f"invalid {self.invalid_count}, voided {self.voided_count}"
                     + ("   COMPLETE" if self.complete else ""))
        return "\n".join(lines)

    def suggested_type(self) -> str | None:
        if self.complete:
            return None
        if not self.real_started:
            for p in PRACTICE_TYPES:
                if not self.practice_done(p):
                    return p
        for c in REAL_TYPES:
            if not self.full(c):
                return c
        return None


# --------------------------------------------------------------------------- CSV log
def rle(seq: Sequence[str]) -> str:
    """Run-length encoding as JSON: ['a','a','b'] -> '[["a",2],["b",1]]' (any label text is safe)."""
    runs: list[list] = []
    for x in seq:
        if runs and runs[-1][0] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    return json.dumps(runs, ensure_ascii=False, separators=(",", ":"))


def unrle(text: str) -> list[str]:
    if not text:
        return []
    return [label for label, n in json.loads(text) for _ in range(int(n))]


CSV_FIELDS = [
    "timestamp", "session_id", "event", "participant", "trial_no", "trial_type", "practice",
    "label_set", "verdict", "reason", "counted", "voided", "void_reason", "duration_s",
    "frames", "error_frames", "max_gap_s", "override_neutral", "quiet_before_start_s",
    "shown_rle", "raw_rle", "gate_db", "frames_to_switch", "sureness",
    "neutral_class", "i1_class", "i2_class", "settings_json",
    "pos_i1", "pos_i2", "pos_acc", "neg_i1", "neg_i2", "neg_acc",
]


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


class TrialLog:
    """Append-only CSV, one file per app session. Rows are never rewritten: a void is a new row.
    If the file cannot be written (for example it is open in Excel), rows wait in `pending`
    and are written with the next row; the trial flow never stops because of the log."""

    def __init__(self, path: Path | str, session_id: str | None = None, defer_until_trial: bool = False):
        self.path = Path(path)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pending: list[dict] = []
        self.last_error: str | None = None
        # with defer_until_trial, rows wait in memory until the first trial, so an app run
        # without trials leaves no file behind
        self.deferring = defer_until_trial

    @classmethod
    def for_session(cls, results_dir: Path | str, session_id: str | None = None,
                    defer_until_trial: bool = False) -> "TrialLog":
        sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(Path(results_dir) / f"trials_{sid}.csv", sid, defer_until_trial)

    def write(self, row: Mapping) -> bool:
        self.pending.append({k: _cell(row.get(k)) for k in CSV_FIELDS})
        if self.deferring and row.get("event") not in ("trial", "void"):
            return True
        self.deferring = False
        return self.flush()

    def flush(self) -> bool:
        if not self.pending:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new:
                    w.writeheader()
                w.writerows(self.pending)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            self.last_error = f"cannot write {self.path.name}: {e}"
            return False
        self.pending.clear()
        self.last_error = None
        return True


def tallies_from_csv(path: Path | str) -> dict[str, dict[str, list[int]]]:
    """Rebuild {participant: {cell: [counted, successes]}} from the log alone (audit / crash recovery)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    trials, out = {}, {}

    def cell(pid, ttype):
        return out.setdefault(pid, {c: [0, 0] for c in REAL_TYPES})[ttype]

    for r in rows:
        key = (r["session_id"], r["trial_no"])
        if r["event"] == "trial":
            trials[key] = r
            if r["counted"] == "1":
                c = cell(r["participant"], r["trial_type"])
                c[0] += 1
                c[1] += r["verdict"] == PASS
        elif r["event"] == "void":
            orig = trials.get(key)
            if orig is not None and orig["counted"] == "1":
                c = cell(orig["participant"], orig["trial_type"])
                c[0] -= 1
                c[1] -= orig["verdict"] == PASS
    return out


# --------------------------------------------------------------------------- session
READY, RECORDING, RESULT = "READY", "RECORDING", "RESULT"


@dataclass(frozen=True)
class Event:
    action: str   # started stopped aborted refused ignored selected voided participant roles
    message: str = ""
    result: TrialResult | None = None

    @property
    def ok(self) -> bool:
        return self.action not in ("refused", "ignored")


class TrialSession:
    """All trial state for one app run. Every method takes `now` (time.monotonic seconds) so the
    logic is testable; the GUI passes time.monotonic(). Thread-safe (one re-entrant lock)."""

    def __init__(self, role_map: RoleMap, *, log: TrialLog | None = None,
                 settings: Mapping | None = None, neutral_hold_s: float = 0.3,
                 debounce_s: float = 0.25, key_stale_s: float = 0.6,
                 audio_timeout_s: float = 0.5, max_gap_s: float = 0.5,
                 invalid_counts_as_fail: bool = False,
                 wall_clock: Callable[[], datetime] = datetime.now):
        self._lock = threading.RLock()
        self.role_map = role_map
        self.log = log
        self.settings = dict(settings or {})
        self.neutral_hold_s = neutral_hold_s
        self.debounce_s = debounce_s
        self.key_stale_s = key_stale_s
        self.audio_timeout_s = audio_timeout_s
        self.max_gap_s = max_gap_s
        self.invalid_counts_as_fail = invalid_counts_as_fail
        self._wall = wall_clock
        self.state = READY
        self.tally: Tally | None = None
        self.history: list[Tally] = []
        self._selected = PRACTICE_NEUTRAL
        self.last_result: TrialResult | None = None
        self.message = ""
        self._trial_no = 0
        self._last_frame_t: float | None = None
        self._last_error: str | None = None  # set while the latest frame was not a classification
        self._neutral_since: float | None = None
        self._key_held = False
        self._key_last_event = float("-inf")
        self._key_last_accept = float("-inf")
        self._clear_window()

    @property
    def selected_type(self) -> str:
        """Read-only: change it with select_type(), which refuses while recording."""
        return self._selected

    # ------------------------------------------------------------------ live inputs
    def on_frame(self, label, now: float, raw: str | None = None, pending: bool = False) -> None:
        """Call once per FRESH audio frame with the label shown on screen for it.
        raw: optional per-frame model guess (logged only). pending: True while the app's
        smoother is building up a tap that is not shown yet (resets the Neutral hold)."""
        with self._lock:
            kind, role = self.role_map.role_of(label)
            self._last_frame_t = now
            if kind == ERROR:
                self._last_error = "no label" if not isinstance(label, str) or not label else label
                self._neutral_since = None
            else:
                self._last_error = None
                if role == NEUTRAL and not pending:
                    if self._neutral_since is None:
                        self._neutral_since = now
                else:
                    self._neutral_since = None
            if self.state != RECORDING:
                return
            self._frame_times.append(now)
            if kind == ERROR:
                self._errors += 1
                self._shown.append(f"<{self._last_error}>")
            else:
                self._labels.add(role)
                self._shown.append(label)
            if raw is not None:
                self._raw.append(str(raw))

    def report_error(self, message: str, now: float) -> None:
        """The app's frame loop raised: the frame was lost. Makes a running trial INVALID
        (unless it is already a definite FAIL) and blocks starts until a good frame arrives."""
        with self._lock:
            self._last_error = f"error: {message}"
            self._neutral_since = None
            if self.state == RECORDING:
                self._errors += 1
                self._shown.append(f"<{self._last_error}>")

    def set_settings(self, settings: Mapping, now: float | None = None) -> None:
        """Gate / frames-to-switch / sureness as now used by the app. A change while recording
        makes that trial INVALID (or FAIL if it already holds a forbidden label)."""
        with self._lock:
            new = dict(settings)
            if self.state == RECORDING and new != self.settings and "settings changed" not in self._tamper:
                self._tamper.append("settings changed")
            self.settings = new

    def set_role_map(self, role_map: RoleMap, now: float) -> Event:
        with self._lock:
            if self.state == RECORDING:
                self._finish(now, abort_reason="roles changed")
            self.role_map = role_map
            self._neutral_since = None
            self._log({"event": "roles", "participant": self.tally.participant if self.tally else "",
                       **self._role_cols(role_map.as_dict())}, now)
            return Event("roles", f"roles: {role_map.as_dict()}")

    # ------------------------------------------------------------------ keys
    def space_down(self, now: float, override: bool = False,
                   text_input_active: bool = False) -> Event:
        """Raw SPACE key-down event (auto-repeats allowed). At most one start/stop per press."""
        with self._lock:
            if text_input_active:
                return Event("ignored", "a text field has the keyboard")
            stale = self._key_held and now - self._key_last_event > self.key_stale_s
            self._key_last_event = now
            if self._key_held and not stale:
                return Event("ignored", "key held down")
            self._key_held = True
            if now - self._key_last_accept < self.debounce_s:
                return Event("ignored", "key bounce")
            ev = self.press(now, override=override)
            if ev.action in ("started", "stopped"):
                self._key_last_accept = now
            return ev

    def space_up(self, now: float | None = None) -> None:
        with self._lock:
            self._key_held = False

    def press(self, now: float, override: bool = False) -> Event:
        """One debounced SPACE press: stop if recording, else try to start."""
        with self._lock:
            if self.state == RECORDING:
                return self._finish(now)
            block = self.start_block(now, override)
            if block is not None:
                self.message = block[1]
                return Event("refused", block[1])
            self._start(now, override)
            return Event("started", self.message)

    # ------------------------------------------------------------------ experimenter actions
    def select_type(self, trial_type: str) -> Event:
        with self._lock:
            if trial_type not in TRIAL_TYPES:
                return Event("refused", f"unknown trial type {trial_type!r}")
            if self.state == RECORDING:
                return Event("refused", "the trial type is locked while recording")
            self._selected = trial_type
            if self.state == RESULT:
                self.state = READY
            return Event("selected", trial_type)

    def abort(self, reason: str, now: float) -> Event:
        """App-initiated stop (mode switch, model load, device change). Never a PASS."""
        with self._lock:
            if self.state != RECORDING:
                return Event("ignored", "not recording")
            return self._finish(now, abort_reason=reason or "aborted")

    def void_last(self, reason: str, now: float) -> Event:
        with self._lock:
            if self.state == RECORDING:
                return Event("refused", "stop the trial first")
            if not reason or not reason.strip():
                return Event("refused", "a reason is required to void a trial")
            if self.tally is None or not self.tally.trials:
                return Event("refused", "no trial to void for this participant")
            last = self.tally.trials[-1]
            if last.voided:
                return Event("refused", f"trial #{last.trial_no} is already voided; only the most recent trial can be voided")
            last.voided, last.void_reason, last.counted = True, reason.strip(), False
            self.state = READY
            self.message = f"voided trial #{last.trial_no} ({last.trial_type}): {last.void_reason}"
            row = self._trial_row(last, "void")
            row["timestamp"] = self._stamp()
            self._log(row, now)
            return Event("voided", self.message, last)

    def next_participant(self, participant: str, now: float, force: bool = False) -> Event:
        with self._lock:
            if self.state == RECORDING:
                return Event("refused", "stop the trial first")
            pid = (participant or "").strip()
            if not pid:
                return Event("refused", "enter a participant ID")
            used = {t.participant.lower() for t in self.history}
            if self.tally is not None:
                used.add(self.tally.participant.lower())
            if pid.lower() in used:
                return Event("refused", f"participant ID {pid!r} was already used in this session")
            old = self.tally
            if old is not None and old.real_started and not old.complete and not force:
                return Event("refused", f"{old.participant} has {old.n_counted_total}/"
                                        f"{TRIALS_PER_PARTICIPANT} counted trials: confirm to leave incomplete")
            if old is not None:
                if old.trials:
                    self._log_summary(old, now)
                self.history.append(old)
            self.tally = Tally(pid)
            self.state = READY
            self.last_result = None
            self._selected = PRACTICE_NEUTRAL
            self._log({"event": "participant", "participant": pid}, now)
            self.message = f"participant {pid}: start with {PRACTICE_NEUTRAL}"
            return Event("participant", self.message)

    def write_summary(self, now: float) -> bool:
        with self._lock:
            return self._log_summary(self.tally, now) if self.tally else False

    # ------------------------------------------------------------------ queries
    def quiet_for(self, now: float) -> float:
        with self._lock:
            if self._neutral_since is None or not self._audio_alive(now):
                return 0.0
            return max(0.0, now - self._neutral_since)

    def start_block(self, now: float, override: bool = False) -> tuple[str, str] | None:
        """(code, message) saying why a start press would be refused now, or None.
        Only code 'not_neutral' can be overridden."""
        with self._lock:
            if self.state == RECORDING:
                return ("recording", "already recording")
            if self.tally is None:
                return ("participant", "enter a participant ID (Next participant) first")
            problems = self.role_map.problems()
            if problems:
                return ("roles", "fix roles: " + "; ".join(problems))
            t, tal = self.selected_type, self.tally
            if t not in TRIAL_TYPES:
                return ("type", "choose a trial type")
            if tal.complete:
                return ("complete", f"COMPLETE: {tal.participant} has all {TRIALS_PER_PARTICIPANT} "
                                    "trials. Enter the sheet, then Next participant")
            if t in PRACTICE_TYPES:
                if tal.real_started:
                    return ("practice", "practice is over: real trials have started")
                if tal.practice_done(t):
                    return ("practice", f"{t} already done (one practice run each). "
                                        f"Next: {tal.suggested_type()}")
            elif tal.full(t):
                return ("cell_full", f"{t} already has {TRIALS_PER_CELL} counted trials. "
                                     f"Next: {tal.suggested_type()}")
            if not self._audio_alive(now):
                return ("no_audio", "NO AUDIO: no fresh microphone frames")
            if self._last_error is not None:
                return ("app_error", f"APP ERROR: {self._last_error}")
            quiet = self.quiet_for(now)
            if not override and quiet < self.neutral_hold_s - 1e-9:
                return ("not_neutral", f"WAIT FOR NEUTRAL ({quiet:.2f} of {self.neutral_hold_s:.2f} s)")
            return None

    def status(self, now: float) -> dict:
        with self._lock:
            block = self.start_block(now) if self.state != RECORDING else None
            tal = self.tally
            return {
                "state": self.state,
                "participant": tal.participant if tal else None,
                "selected_type": self.selected_type,
                "locked_type": self._type if self.state == RECORDING else None,
                "elapsed_s": now - self._t_start if self.state == RECORDING else None,
                "frames": len(self._frame_times) if self.state == RECORDING else None,
                "start_block": block,
                "can_override": block is not None and block[0] == "not_neutral",
                "quiet_for_s": self.quiet_for(now),
                "cells": {c: (tal.n_counted(c), tal.successes(c)) for c in REAL_TYPES} if tal else {},
                "complete": bool(tal and tal.complete),
                "missing_practice": tal.missing_practice() if tal else [],
                "suggested_type": tal.suggested_type() if tal else None,
                "last_result": self.last_result.display_text() if self.last_result else None,
                "message": self.message,
                "log_error": self.log.last_error if self.log else None,
                "log_pending": len(self.log.pending) if self.log else 0,
                "role_warnings": self.role_map.warnings(),
            }

    # ------------------------------------------------------------------ internals
    def _audio_alive(self, now: float) -> bool:
        return self._last_frame_t is not None and now - self._last_frame_t <= self.audio_timeout_s

    def _clear_window(self):
        self._type: str | None = None
        self._t_start = 0.0
        self._labels: set = set()
        self._shown: list = []
        self._raw: list = []
        self._frame_times: list = []
        self._errors = 0
        self._tamper: list = []
        self._override = False
        self._quiet0 = 0.0
        self._settings0: dict = {}
        self._roles0: dict = {}

    def _start(self, now: float, override: bool):
        self._clear_window()
        self._trial_no += 1
        quiet = self.quiet_for(now)
        self._type = self.selected_type
        self._t_start = now
        self._override = bool(override) and quiet < self.neutral_hold_s - 1e-9
        self._quiet0 = quiet
        self._settings0 = dict(self.settings)
        self._roles0 = self.role_map.as_dict()
        self.state = RECORDING
        self.message = f"RECORDING #{self._trial_no} {self._type}" + \
                       ("  (started with override, display not Neutral)" if self._override else "")

    def _finish(self, now: float, abort_reason: str | None = None) -> Event:
        times = [self._t_start] + self._frame_times + [now]
        max_gap = max(b - a for a, b in zip(times, times[1:]))
        extra = list(self._tamper)
        if abort_reason:
            extra.append(f"aborted: {abort_reason}")
        v, reason = judge(self._type, self._labels, frames=len(self._frame_times),
                          error_frames=self._errors, max_gap_s=max_gap,
                          gap_limit_s=self.max_gap_s, extra_invalid=extra)
        counted = self._type in REAL_TYPES and (
            v in (PASS, FAIL) or (v == INVALID and self.invalid_counts_as_fail))
        res = TrialResult(
            trial_no=self._trial_no, participant=self.tally.participant, trial_type=self._type,
            labels=frozenset(self._labels), verdict=v, reason=reason, counted=counted,
            t_start=self._t_start, t_stop=now, frames=len(self._frame_times),
            error_frames=self._errors, max_gap_s=max_gap, override_neutral=self._override,
            quiet_before_start_s=self._quiet0, shown_seq=list(self._shown), raw_seq=list(self._raw),
            settings=dict(self._settings0), roles=dict(self._roles0), timestamp=self._stamp())
        self.tally.trials.append(res)
        self.last_result = res
        self.state = RESULT
        self.message = res.display_text()
        if self.tally.complete:
            self.message += "   COMPLETE: enter the sheet, then Next participant"
        self._log(self._trial_row(res, "trial"), now)
        self._clear_window()
        return Event("aborted" if abort_reason else "stopped", self.message, res)

    def _stamp(self) -> str:
        return self._wall().isoformat(timespec="milliseconds")

    @staticmethod
    def _role_cols(roles: Mapping) -> dict:
        return {"neutral_class": roles.get("neutral"), "i1_class": roles.get("i1"),
                "i2_class": roles.get("i2")}

    def _trial_row(self, r: TrialResult, event: str) -> dict:
        s = r.settings
        return {
            "timestamp": r.timestamp, "event": event, "participant": r.participant,
            "trial_no": r.trial_no, "trial_type": r.trial_type, "practice": r.practice,
            "label_set": r.set_text, "verdict": r.verdict, "reason": r.reason,
            "counted": r.counted, "voided": r.voided, "void_reason": r.void_reason,
            "duration_s": f"{r.duration_s:.3f}", "frames": r.frames,
            "error_frames": r.error_frames, "max_gap_s": f"{r.max_gap_s:.3f}",
            "override_neutral": r.override_neutral,
            "quiet_before_start_s": f"{r.quiet_before_start_s:.2f}",
            "shown_rle": rle(r.shown_seq), "raw_rle": rle(r.raw_seq) if r.raw_seq else "",
            "gate_db": s.get("gate_db"), "frames_to_switch": s.get("frames_to_switch"),
            "sureness": s.get("sureness"),
            "settings_json": json.dumps(s, sort_keys=True, default=str),
            **self._role_cols(r.roles),
        }

    def _log_summary(self, tal: Tally, now: float) -> bool:
        (_, p1, p2, pa), (_, n1, n2, na) = tal.sheet_rows()
        return self._log({
            "event": "summary", "participant": tal.participant,
            "counted": tal.n_counted_total, "voided": tal.voided_count,
            "reason": f"invalid {tal.invalid_count}" + ("; COMPLETE" if tal.complete else "; incomplete"),
            "pos_i1": p1, "pos_i2": p2, "pos_acc": f"{pa:.1f}",
            "neg_i1": n1, "neg_i2": n2, "neg_acc": f"{na:.1f}",
        }, now)

    def _log(self, row: dict, now: float) -> bool:
        if self.log is None:
            return True
        row = dict(row)
        row.setdefault("timestamp", self._stamp())
        row["session_id"] = self.log.session_id
        return self.log.write(row)
