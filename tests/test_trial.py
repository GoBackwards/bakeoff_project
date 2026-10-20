"""Unit tests for trial.py. Run from the repo root: uv run python -m unittest tests.test_trial   (standard library only)."""
import csv
import itertools
import random
import sys
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
import trial  # noqa: E402
from trial import (  # noqa: E402
    NEUTRAL as N, I1, I2, ROLES, POS_I1, POS_I2, NEG_I1, NEG_I2, PRACTICE_NEUTRAL,
    PRACTICE_I1, PRACTICE_I2, REAL_TYPES, PRACTICE_TYPES, PASS, FAIL, INVALID, SET_ONLY,
    READY, RECORDING, RESULT, CSV_FIELDS, verdict, definitely_fails, format_set, judge,
    detect_neutral, RoleMap, TrialSession, TrialLog, rle, unrle, tallies_from_csv,
)

CLASSES = ["neutral", "bottle", "bowl"]
DT = 1 / 44  # the app's fresh-frame rate
SETTINGS = {"gate_db": 12.5, "frames_to_switch": 5, "sureness": 0.5}
X = "Unmapped(typing)"


def fixed_clock():
    return datetime(2026, 10, 22, 14, 0, 0)


class Rig:
    """A session with a fake clock and a temp CSV."""

    def __init__(self, classes=CLASSES, participant="p1", log=True, **kw):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = 100.0
        self.log = TrialLog(Path(self.tmp.name) / "trials.csv", "S1") if log else None
        self.s = TrialSession(RoleMap.from_classes(classes), log=self.log, settings=SETTINGS,
                              wall_clock=fixed_clock, **kw)
        if participant:
            assert self.s.next_participant(participant, self.t).ok

    def close(self):
        self.tmp.cleanup()

    def frames(self, label, n, **kw):
        for _ in range(n):
            self.t += DT
            self.s.on_frame(label, self.t, **kw)

    def space(self, override=False):
        self.t += 0.001
        ev = self.s.space_down(self.t, override=override)
        self.s.space_up(self.t)
        return ev

    def rows(self):
        with open(self.log.path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def trial(self, ttype, seq, pre_quiet=15):
        """Select, wait for Neutral, SPACE, feed `seq` [(label, n)...], SPACE. Returns the result."""
        assert self.s.select_type(ttype).ok
        self.frames("neutral", pre_quiet)
        ev = self.space()
        assert ev.action == "started", ev.message
        for label, n in seq:
            self.frames(label, n)
        ev = self.space()
        assert ev.action == "stopped", ev.message
        return ev.result


TAP_BOTTLE = [("neutral", 5), ("bottle", 20), ("neutral", 8)]
TAP_BOWL = [("neutral", 5), ("bowl", 20), ("neutral", 8)]
QUIET = [("neutral", 33)]


def subsets(universe):
    u = list(universe)
    return [frozenset(c) for r in range(len(u) + 1) for c in itertools.combinations(u, r)]


# =========================================================================== verdict
class TestVerdict(unittest.TestCase):
    # Hand-typed from the slides, independent of PASS_SETS.
    EXPECTED = {
        #  set                  PosI1 PosI2 NegI1 NegI2
        frozenset():            (0, 0, 0, 0),
        frozenset({N}):         (0, 0, 1, 1),
        frozenset({I1}):        (1, 0, 0, 1),
        frozenset({I2}):        (0, 1, 1, 0),
        frozenset({N, I1}):     (1, 0, 0, 1),
        frozenset({N, I2}):     (0, 1, 1, 0),
        frozenset({I1, I2}):    (0, 0, 0, 0),
        frozenset({N, I1, I2}): (0, 0, 0, 0),
    }

    def test_exhaustive_table(self):
        self.assertEqual(set(subsets(ROLES)), set(self.EXPECTED))
        for s, exp in self.EXPECTED.items():
            got = tuple(int(verdict(t, s)) for t in REAL_TYPES)
            self.assertEqual(got, exp, format_set(s))

    def test_practice_judged_like_positive(self):
        for s in subsets(ROLES):
            self.assertEqual(verdict(PRACTICE_I1, s), verdict(POS_I1, s))
            self.assertEqual(verdict(PRACTICE_I2, s), verdict(POS_I2, s))

    def test_unknown_labels_always_fail(self):
        for base in subsets(ROLES):
            for junk in ({X}, {"garbage"}, {X, "garbage"}):
                for t in REAL_TYPES:
                    self.assertFalse(verdict(t, base | junk), (t, base | junk))
                    self.assertTrue(definitely_fails(t, base | junk))

    def test_iterables_and_duplicates(self):
        self.assertTrue(verdict(POS_I1, [N, I1, I1, N]))
        self.assertTrue(verdict(NEG_I2, (I1,)))
        self.assertTrue(verdict(NEG_I1, iter([N, N])))

    def test_bad_types_raise(self):
        for bad in ("Pos I3", PRACTICE_NEUTRAL, None):
            with self.assertRaises(ValueError):
                verdict(bad, {N})
            with self.assertRaises(ValueError):
                definitely_fails(bad, {N})

    def test_rule_equivalence(self):
        """PASS iff the set is non-empty, uses only allowed labels, and a positive trial shows its target."""
        allowed = {POS_I1: {N, I1}, POS_I2: {N, I2}, NEG_I1: {N, I2}, NEG_I2: {N, I1}}
        target = {POS_I1: I1, POS_I2: I2}
        for s in subsets(ROLES + ("x",)):
            for t in REAL_TYPES:
                rule = bool(s) and s <= allowed[t] and (t not in target or target[t] in s)
                self.assertEqual(verdict(t, s), rule, (t, s))

    def test_definitely_fails_is_exact(self):
        """definitely_fails(S) iff no superset of S (more frames) could pass."""
        universe = subsets(ROLES + ("x",))
        for t in REAL_TYPES:
            for s in universe:
                some_superset_passes = any(verdict(t, sup) for sup in universe if s <= sup)
                self.assertEqual(definitely_fails(t, s), not some_superset_passes, (t, s))

    def test_format_set(self):
        self.assertEqual(format_set({I2, N}), "{Neutral, Interaction#2}")
        self.assertEqual(format_set(set()), "{}")
        self.assertEqual(format_set({X, I1, N, I2}),
                         "{Neutral, Interaction#1, Interaction#2, Unmapped(typing)}")


class TestJudge(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(judge(POS_I1, {N, I1}, frames=40), (PASS, ""))
        self.assertEqual(judge(POS_I1, {N}, frames=40), (FAIL, "Interaction#1 never shown"))
        self.assertEqual(judge(NEG_I1, {N, I1}, frames=40)[0], FAIL)
        self.assertEqual(judge(NEG_I1, set(), frames=3), (FAIL, "empty set"))  # inconsistent input still fails

    def test_invalid_conditions(self):
        self.assertEqual(judge(NEG_I1, set(), frames=0)[0], INVALID)
        self.assertEqual(judge(NEG_I1, {N}, frames=40, error_frames=1)[0], INVALID)
        self.assertEqual(judge(NEG_I1, {N}, frames=40, max_gap_s=0.8)[0], INVALID)
        self.assertEqual(judge(NEG_I1, {N}, frames=40, max_gap_s=0.5)[0], PASS)  # limit is inclusive
        self.assertEqual(judge(NEG_I1, {N}, frames=40, extra_invalid=["aborted: x"])[0], INVALID)
        self.assertEqual(judge(POS_I1, {N, I1}, frames=40, error_frames=2)[0], INVALID)

    def test_definite_fail_beats_invalid(self):
        v, reason = judge(POS_I1, {N, I2}, frames=40, error_frames=1)
        self.assertEqual(v, FAIL)
        self.assertIn("Interaction#2", reason)
        self.assertIn("also", reason)
        self.assertEqual(judge(NEG_I2, {I1, N}, frames=40, max_gap_s=3)[0], INVALID)  # I1 is allowed in Neg I2
        self.assertEqual(judge(NEG_I2, {I2}, frames=40, max_gap_s=3)[0], FAIL)

    def test_practice_neutral(self):
        self.assertEqual(judge(PRACTICE_NEUTRAL, {N, I1}, frames=10)[0], SET_ONLY)
        self.assertEqual(judge(PRACTICE_NEUTRAL, set(), frames=0)[0], INVALID)
        self.assertEqual(judge(PRACTICE_I1, {I1}, frames=10)[0], PASS)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            judge("Neg I3", {N}, frames=1)

    def test_never_pass_when_invalid_random(self):
        rng = random.Random(209)
        labels = list(ROLES) + [X]
        for _ in range(5000):
            t = rng.choice(REAL_TYPES + (PRACTICE_I1, PRACTICE_I2))
            s = {x for x in labels if rng.random() < 0.4}
            frames = rng.choice([0, 1, 30])
            err = rng.choice([0, 0, 1, 3])
            gap = rng.choice([0.02, 0.4, 0.6, 2.0])
            extra = rng.choice([[], [], ["settings changed"], ["aborted: mode"]])
            v, _ = judge(t, s, frames=frames, error_frames=err, max_gap_s=gap, extra_invalid=extra)
            invalid = frames == 0 or err > 0 or (frames > 0 and gap > 0.5) or extra
            if invalid:
                self.assertNotEqual(v, PASS)
            if v == PASS:
                self.assertTrue(verdict(t, s))


# =========================================================================== roles
class TestRoleMap(unittest.TestCase):
    def test_defaults(self):
        rm = RoleMap.from_classes(CLASSES)
        self.assertEqual((rm.neutral, rm.i1, rm.i2), ("neutral", "bottle", "bowl"))
        self.assertEqual(rm.problems(), [])
        self.assertEqual(rm.warnings(), [])

    def test_default_words_and_case(self):
        rm = RoleMap.from_classes(["Silence", "steel_bottle", "Ceramic Bowl"])
        self.assertEqual((rm.neutral, rm.i1, rm.i2), ("Silence", "steel_bottle", "Ceramic Bowl"))

    def test_detect_neutral_matches_app_rule(self):
        self.assertEqual(detect_neutral(["bottle", "Neutral", "bowl"]), "Neutral")
        self.assertEqual(detect_neutral(["bottle", "background_noise", "bowl"]), "background_noise")
        self.assertEqual(detect_neutral(["bottle", "bowl"]), "Neutral")  # app's smoother shows "Neutral"
        self.assertEqual(detect_neutral(["quiet room", "neutral"]), "neutral")  # exact name wins

    def test_explicit_choice_wins(self):
        rm = RoleMap.from_classes(CLASSES, i1="bowl", i2="bottle")
        self.assertEqual((rm.i1, rm.i2), ("bowl", "bottle"))
        self.assertEqual(rm.role_of("bowl"), ("role", I1))

    def test_no_order_based_guess(self):
        rm = RoleMap.from_classes(["neutral", "jar", "cup"])
        self.assertIsNone(rm.i1)
        self.assertIsNone(rm.i2)
        self.assertEqual(len(rm.problems()), 2)
        amb = RoleMap.from_classes(["neutral", "bottle_light", "bottle_hard", "bowl"])
        self.assertIsNone(amb.i1)
        self.assertEqual(amb.i2, "bowl")

    def test_problems(self):
        self.assertTrue(any("same class" in p for p in RoleMap(tuple(CLASSES), "neutral", "bowl", "bowl").problems()))
        self.assertTrue(any("Neutral class" in p for p in RoleMap(tuple(CLASSES), "neutral", "neutral", "bowl").problems()))
        self.assertTrue(any("not in the model" in p for p in RoleMap(tuple(CLASSES), "neutral", "jar", "bowl").problems()))

    def test_warnings(self):
        rm = RoleMap.from_classes(["neutral", "bottle", "bowl", "typing"])
        self.assertEqual(rm.problems(), [])
        self.assertTrue(any("typing" in w for w in rm.warnings()))
        self.assertTrue(any("no Neutral class" in w for w in RoleMap.from_classes(["bottle", "bowl"]).warnings()))

    def test_role_of(self):
        rm = RoleMap.from_classes(["neutral", "bottle", "bowl", "typing"])
        self.assertEqual(rm.role_of("neutral"), ("role", N))
        self.assertEqual(rm.role_of("bottle"), ("role", I1))
        self.assertEqual(rm.role_of("bowl"), ("role", I2))
        self.assertEqual(rm.role_of("typing"), ("unmapped", X))
        for status_text in ("press Calibrate Quiet", "listening...", "model does not match this microphone: retrain",
                            "", None, 3, "Bottle"):
            self.assertEqual(rm.role_of(status_text), ("error", None), status_text)
        no_neutral = RoleMap.from_classes(["bottle", "bowl"])
        self.assertEqual(no_neutral.role_of("Neutral"), ("role", N))


# =========================================================================== keys
class TestKeys(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        self.r.frames("neutral", 15)

    def tearDown(self):
        self.r.close()

    def test_held_key_and_bounce(self):
        s, t = self.r.s, self.r.t
        self.assertEqual(s.space_down(t).action, "started")
        self.assertEqual(s.space_down(t + 0.05).action, "ignored")   # auto-repeat while held
        s.space_up(t + 0.06)
        self.assertEqual(s.space_down(t + 0.10).action, "ignored")   # second press inside debounce
        s.space_up(t + 0.11)
        self.assertEqual(s.state, RECORDING)
        self.assertEqual(s.space_down(t + 0.30).action, "stopped")
        self.assertEqual(s.space_down(t + 0.35).action, "ignored")   # still held: no new start
        self.assertEqual(s.state, RESULT)

    def test_text_field_focus(self):
        ev = self.r.s.space_down(self.r.t, text_input_active=True)
        self.assertEqual(ev.action, "ignored")
        self.assertEqual(self.r.s.space_down(self.r.t + 0.01).action, "started")  # not latched

    def test_lost_release_recovers(self):
        s, t = self.r.s, self.r.t
        self.assertEqual(s.space_down(t).action, "started")
        # release event lost (focus change); next press 1.5 s later still works
        self.r.frames("neutral", 66)
        self.assertEqual(s.space_down(self.r.t).action, "stopped")

    def test_refused_press_does_not_consume_debounce(self):
        s = self.r.s
        s.select_type(POS_I1)
        self.r.frames("bowl", 3)
        self.r.frames("neutral", 13)                 # Neutral for ~0.27 s: too early
        self.assertEqual(self.r.space().action, "refused")
        t_refused = self.r.t
        self.r.frames("neutral", 2)
        self.assertEqual(self.r.space().action, "started")
        self.assertLess(self.r.t - t_refused, self.r.s.debounce_s)


# =========================================================================== state machine
class TestStateMachine(unittest.TestCase):
    def setUp(self):
        self.r = Rig()

    def tearDown(self):
        self.r.close()

    def test_cycle_and_space_in_result_starts_next(self):
        s = self.r.s
        self.assertEqual(s.state, READY)
        self.r.trial(PRACTICE_NEUTRAL, QUIET)
        self.assertEqual(s.state, RESULT)
        s.select_type(PRACTICE_I1)
        self.assertEqual(s.state, READY)
        self.r.trial(POS_I1, TAP_BOTTLE)
        self.assertEqual(s.state, RESULT)
        self.r.frames("neutral", 15)               # type unchanged, still RESULT: SPACE starts the next
        self.assertEqual(s.state, RESULT)
        self.assertEqual(self.r.space().action, "started")
        self.assertEqual(s.state, RECORDING)
        with self.assertRaises(AttributeError):    # no way around select_type()
            s.selected_type = NEG_I2

    def test_type_locked_at_start(self):
        s = self.r.s
        for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
            self.r.trial(p, seq)
        s.select_type(NEG_I1)
        self.r.frames("neutral", 15)
        self.assertEqual(self.r.space().action, "started")
        ev = s.select_type(POS_I2)
        self.assertEqual(ev.action, "refused")
        self.assertEqual(s.selected_type, NEG_I1)
        self.assertEqual(s.status(self.r.t)["locked_type"], NEG_I1)
        self.r.frames("bowl", 20)
        res = self.r.space().result
        self.assertEqual(res.trial_type, NEG_I1)
        self.assertEqual(res.verdict, PASS)

    def test_unknown_type_refused(self):
        self.assertEqual(self.r.s.select_type("Positive I3").action, "refused")

    def test_needs_participant(self):
        r = Rig(participant=None)
        try:
            r.frames("neutral", 15)
            ev = r.space()
            self.assertEqual(ev.action, "refused")
            self.assertEqual(r.s.start_block(r.t)[0], "participant")
        finally:
            r.close()

    def test_role_problems_block(self):
        r = Rig(classes=["neutral", "jar", "cup"])
        try:
            r.frames("neutral", 15)
            self.assertEqual(r.s.start_block(r.t)[0], "roles")
            self.assertEqual(r.space(override=True).action, "refused")
            r.s.set_role_map(RoleMap.from_classes(["neutral", "jar", "cup"], i1="jar", i2="cup"), r.t)
            r.frames("neutral", 15)
            self.assertEqual(r.space().action, "started")
        finally:
            r.close()


# =========================================================================== Neutral hold
class TestNeutralHold(unittest.TestCase):
    def setUp(self):
        self.r = Rig()

    def tearDown(self):
        self.r.close()

    def test_hold_required(self):
        r = self.r
        r.frames("neutral", 6)                      # ~0.14 s
        ev = r.space()
        self.assertEqual(ev.action, "refused")
        self.assertIn("WAIT FOR NEUTRAL", ev.message)
        self.assertTrue(r.s.status(r.t)["can_override"])
        r.frames("neutral", 9)                      # now ~0.32 s since the first Neutral frame
        self.assertGreaterEqual(r.s.quiet_for(r.t), 0.3 - 1e-9)
        self.assertEqual(r.space().action, "started")

    def test_tap_on_screen_resets_hold(self):
        r = self.r
        r.frames("neutral", 20)
        r.frames("bowl", 10)
        self.assertEqual(r.space().action, "refused")
        r.frames("neutral", 10)
        self.assertEqual(r.space().action, "refused")
        r.frames("neutral", 5)
        self.assertEqual(r.space().action, "started")

    def test_pending_tap_resets_hold(self):
        r = self.r
        r.frames("neutral", 20)
        r.frames("neutral", 2, pending=True)        # smoother is building a tap not yet shown
        self.assertEqual(r.space().action, "refused")
        self.assertEqual(r.s.quiet_for(r.t), 0.0)

    def test_override(self):
        r = self.r
        r.frames("bottle", 10)
        ev = r.space(override=True)
        self.assertEqual(ev.action, "started")
        self.assertIn("override", ev.message)
        r.frames("bottle", 15)
        res = r.space().result
        self.assertTrue(res.override_neutral)
        self.assertEqual(r.rows()[-1]["override_neutral"], "1")

    def test_override_flag_only_when_used(self):
        r = self.r
        r.frames("neutral", 15)
        r.space(override=True)
        r.frames("neutral", 15)
        self.assertFalse(r.space().result.override_neutral)

    def test_no_audio_blocks_even_with_override(self):
        r = self.r
        self.assertEqual(r.s.start_block(r.t, override=True)[0], "no_audio")   # no frame yet
        r.frames("neutral", 20)
        r.t += 0.6                                                               # stream stalled
        self.assertEqual(r.space(override=True).action, "refused")
        self.assertEqual(r.s.start_block(r.t, override=True)[0], "no_audio")
        self.assertEqual(r.s.quiet_for(r.t), 0.0)

    def test_app_error_blocks_even_with_override(self):
        r = self.r
        r.frames("neutral", 20)
        r.frames("press Calibrate Quiet", 1)
        self.assertEqual(r.s.start_block(r.t, override=True)[0], "app_error")
        r.frames("neutral", 1)
        r.s.report_error("boom", r.t)
        self.assertEqual(r.s.start_block(r.t, override=True)[0], "app_error")
        r.frames("neutral", 15)
        self.assertIsNone(r.s.start_block(r.t))


# =========================================================================== window
class TestWindow(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
            self.r.trial(p, seq)

    def tearDown(self):
        self.r.close()

    def test_only_frames_between_presses(self):
        r = self.r
        r.frames("bowl", 20)                          # before: not in the set
        res = r.trial(NEG_I2, [("neutral", 20)])
        self.assertEqual(res.labels, frozenset({N}))
        r.frames("bottle", 20)                        # right after stop: no grace tail
        self.assertEqual(res.labels, frozenset({N}))
        self.assertEqual(res.frames, 20)
        self.assertEqual(r.s.tally.trials[-1].frames, 20)

    def test_tap_right_after_stop_not_counted(self):
        r = self.r
        res = r.trial(POS_I1, [("neutral", 20)])
        r.frames("bottle", 25)
        self.assertEqual(res.verdict, FAIL)
        self.assertEqual(res.labels, frozenset({N}))

    def test_no_minimum_duration(self):
        r = self.r
        r.s.select_type(NEG_I1)
        r.frames("neutral", 15)
        self.assertEqual(r.space().action, "started")
        r.frames("neutral", 11)                       # 0.25 s + 1 ms press offset
        ev = r.space()
        self.assertEqual(ev.action, "stopped")
        self.assertLess(ev.result.duration_s, 0.3)
        self.assertEqual(ev.result.verdict, PASS)

    def test_duration_and_seq(self):
        r = self.r
        res = r.trial(POS_I2, TAP_BOWL)
        self.assertAlmostEqual(res.duration_s, 33 * DT + 0.001, places=6)
        self.assertEqual(res.shown_seq, ["neutral"] * 5 + ["bowl"] * 20 + ["neutral"] * 8)
        self.assertEqual(res.set_text, "{Neutral, Interaction#2}")
        self.assertEqual(res.verdict, PASS)


# =========================================================================== INVALID
class TestInvalid(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
            self.r.trial(p, seq)

    def tearDown(self):
        self.r.close()

    def start(self, ttype):
        self.r.s.select_type(ttype)
        self.r.frames("neutral", 15)
        self.assertEqual(self.r.space().action, "started")

    def test_no_fresh_frames(self):
        r = self.r
        self.start(NEG_I1)
        r.t += 0.4
        res = r.space().result
        self.assertEqual(res.verdict, INVALID)
        self.assertIn("no fresh audio", res.reason)
        self.assertFalse(res.counted)
        self.assertEqual(r.s.tally.n_counted(NEG_I1), 0)

    def test_gap(self):
        r = self.r
        self.start(NEG_I1)
        r.frames("neutral", 5)
        r.t += 0.7
        r.frames("neutral", 5)
        res = r.space().result
        self.assertEqual(res.verdict, INVALID)
        self.assertIn("audio gap", res.reason)

    def test_error_label_and_reported_error(self):
        r = self.r
        self.start(NEG_I1)
        r.frames("neutral", 5)
        r.frames("press Calibrate Quiet", 1)
        r.frames("neutral", 10)
        res = r.space().result
        self.assertEqual(res.verdict, INVALID)
        self.assertEqual(res.error_frames, 1)
        self.assertIn("<press Calibrate Quiet>", res.shown_seq)
        self.start(NEG_I1)
        r.frames("neutral", 6)
        r.s.report_error("ValueError: shapes", r.t)
        r.frames("neutral", 8)
        self.assertEqual(r.space().result.verdict, INVALID)

    def test_settings_change(self):
        r = self.r
        self.start(NEG_I2)
        r.s.set_settings(SETTINGS, r.t)              # same values: no effect
        r.frames("neutral", 15)
        self.assertEqual(r.space().result.verdict, PASS)
        self.start(NEG_I2)
        r.s.set_settings({**SETTINGS, "sureness": 1.0}, r.t)
        r.frames("neutral", 15)
        res = r.space().result
        self.assertEqual(res.verdict, INVALID)
        self.assertEqual(res.settings["sureness"], 0.5)   # logged settings = those at start

    def test_abort_and_roles_change(self):
        r = self.r
        self.start(POS_I1)
        r.frames("neutral", 5)
        ev = r.s.abort("mode switched to Train", r.t)
        self.assertEqual(ev.action, "aborted")
        self.assertEqual(ev.result.verdict, INVALID)
        self.assertEqual(r.s.state, RESULT)
        self.assertEqual(r.s.abort("again", r.t).action, "ignored")
        self.start(POS_I1)
        r.frames("bottle", 15)
        ev = r.s.set_role_map(RoleMap.from_classes(CLASSES, i1="bowl", i2="bottle"), r.t)
        last = r.s.tally.trials[-1]
        self.assertEqual(last.verdict, INVALID)
        self.assertIn("roles changed", last.reason)

    def test_definite_fail_beats_invalid(self):
        r = self.r
        self.start(POS_I1)
        r.frames("bowl", 20)                         # forbidden in Pos I1
        r.frames("press Calibrate Quiet", 1)
        res = r.space().result
        self.assertEqual(res.verdict, FAIL)
        self.assertTrue(res.counted)
        self.start(NEG_I1)
        r.frames("bottle", 20)                       # forbidden in Neg I1
        r.t += 1.0
        r.frames("neutral", 3)
        res = r.space().result
        self.assertEqual(res.verdict, FAIL)
        self.assertTrue(res.counted)
        self.start(NEG_I2)
        r.frames("bowl", 20)                         # forbidden in Neg I2 ...
        ev = r.s.abort("model loaded", r.t)          # ... so aborting cannot hide it
        self.assertEqual(ev.result.verdict, FAIL)
        self.assertTrue(ev.result.counted)

    def test_invalid_counts_as_fail_option(self):
        r = Rig(invalid_counts_as_fail=True)
        try:
            for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
                r.trial(p, seq)
            r.s.select_type(NEG_I1)
            r.frames("neutral", 15)
            r.space()
            r.t += 0.4
            res = r.space().result
            self.assertEqual(res.verdict, INVALID)
            self.assertTrue(res.counted)
            self.assertFalse(res.success)
            self.assertEqual((r.s.tally.n_counted(NEG_I1), r.s.tally.successes(NEG_I1)), (1, 0))
        finally:
            r.close()

    def test_random_streams_never_pass_when_invalid(self):
        rng = random.Random(7)
        r = self.r
        labels = ["neutral", "bottle", "bowl"]
        seen = set()
        for i in range(300):
            open_cells = [c for c in REAL_TYPES if not r.s.tally.full(c)]
            if not open_cells:
                self.assertTrue(r.s.next_participant(f"auto{i}", r.t).ok)
                continue
            r.s.select_type(rng.choice(open_cells))
            r.frames("neutral", 15)
            ev = r.space()
            self.assertEqual(ev.action, "started", ev.message)
            fault = rng.choice(["none", "none", "error", "gap", "settings"])
            for _ in range(rng.randint(1, 4)):
                r.frames(rng.choice(labels), rng.randint(3, 12))
            if fault == "error":
                r.frames("listening...", 1)
            elif fault == "gap":
                r.t += 0.8
                r.frames("neutral", 1)
            elif fault == "settings":
                r.s.set_settings({**SETTINGS, "gate_db": rng.random()}, r.t)
            r.frames("neutral", 12)
            res = r.space().result
            r.s.set_settings(SETTINGS, r.t)
            seen.add((fault, res.verdict))
            if fault != "none":
                self.assertNotEqual(res.verdict, PASS, (fault, res.display_text()))
            if res.verdict == PASS:
                self.assertTrue(verdict(res.trial_type, res.labels))
            self.assertEqual(res.counted, res.verdict in (PASS, FAIL))
        for fault in ("error", "gap", "settings"):          # both outcomes really occurred
            self.assertIn((fault, INVALID), seen)
            self.assertIn((fault, FAIL), seen)
        self.assertIn(("none", PASS), seen)


# =========================================================================== practice
class TestPractice(unittest.TestCase):
    def setUp(self):
        self.r = Rig()

    def tearDown(self):
        self.r.close()

    def test_practice_flow(self):
        r, s = self.r, self.r.s
        self.assertEqual(s.selected_type, PRACTICE_NEUTRAL)
        self.assertEqual(s.tally.suggested_type(), PRACTICE_NEUTRAL)
        res = r.trial(PRACTICE_NEUTRAL, TAP_BOWL)        # neutral practice shows the set only
        self.assertEqual(res.verdict, SET_ONLY)
        self.assertFalse(res.counted)
        self.assertIn("{Neutral, Interaction#2}", res.display_text())
        self.assertEqual(s.tally.suggested_type(), PRACTICE_I1)
        res = r.trial(PRACTICE_I1, TAP_BOTTLE)
        self.assertEqual((res.verdict, res.counted), (PASS, False))
        res = r.trial(PRACTICE_I2, TAP_BOTTLE)
        self.assertEqual((res.verdict, res.counted), (FAIL, False))
        self.assertEqual(s.tally.n_counted_total, 0)
        self.assertEqual(s.tally.suggested_type(), POS_I1)

    def test_one_practice_each(self):
        r, s = self.r, self.r.s
        r.trial(PRACTICE_I1, TAP_BOTTLE)
        s.select_type(PRACTICE_I1)
        r.frames("neutral", 15)
        ev = r.space()
        self.assertEqual(ev.action, "refused")
        self.assertIn("one practice run each", ev.message)

    def test_invalid_or_voided_practice_can_be_redone(self):
        r, s = self.r, self.r.s
        s.select_type(PRACTICE_I1)
        r.frames("neutral", 15)
        r.space()
        r.t += 0.4
        self.assertEqual(r.space().result.verdict, INVALID)
        r.trial(PRACTICE_I1, TAP_BOTTLE)
        self.assertTrue(s.void_last("participant was not ready", r.t).ok)
        r.trial(PRACTICE_I1, TAP_BOTTLE)
        self.assertTrue(s.tally.practice_done(PRACTICE_I1))

    def test_practice_closed_after_real_trials(self):
        r, s = self.r, self.r.s
        r.trial(POS_I1, TAP_BOTTLE)
        s.select_type(PRACTICE_NEUTRAL)
        r.frames("neutral", 15)
        ev = r.space()
        self.assertEqual(ev.action, "refused")
        self.assertIn("practice is over", ev.message)
        self.assertEqual(s.tally.missing_practice(), list(PRACTICE_TYPES))
        self.assertEqual(s.tally.suggested_type(), POS_I1)


# =========================================================================== tally
def run_block(r, ttype, outcomes):
    """outcomes: list of 1/0 (pass/fail) for this cell, using realistic displayed streams."""
    good = {POS_I1: TAP_BOTTLE, POS_I2: TAP_BOWL, NEG_I1: TAP_BOWL, NEG_I2: QUIET}
    bad = {POS_I1: QUIET, POS_I2: TAP_BOTTLE, NEG_I1: TAP_BOTTLE, NEG_I2: TAP_BOWL}
    return [r.trial(ttype, good[ttype] if ok else bad[ttype]) for ok in outcomes]


class TestTally(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
            self.r.trial(p, seq)

    def tearDown(self):
        self.r.close()

    def test_slide_example(self):
        r, tal = self.r, self.r.s.tally
        run_block(r, POS_I1, [1, 1, 1, 0, 0])
        self.assertEqual(tal.positive_accuracy(), 0.3)    # successes / 10 even while incomplete
        run_block(r, POS_I2, [1, 1, 1, 1, 0])
        run_block(r, NEG_I1, [1] * 5)
        run_block(r, NEG_I2, [1] * 5)
        self.assertEqual(tal.sheet_rows(), [
            ["Successful trials for Positive", 3, 4, 0.7],
            ["Successful trials for Negative", 5, 5, 1.0],
        ])
        self.assertTrue(tal.complete)
        self.assertIn("Successful trials for Positive: 3, 4, 0.7", tal.sheet_text())
        self.assertIn("COMPLETE", r.s.message)

    def test_sixth_refused_and_complete(self):
        r, s = self.r, self.r.s
        run_block(r, POS_I1, [1] * 5)
        s.select_type(POS_I1)
        r.frames("neutral", 15)
        ev = r.space()
        self.assertEqual(ev.action, "refused")
        self.assertEqual(s.start_block(r.t)[0], "cell_full")
        self.assertIn("Next: Positive I2", ev.message)
        self.assertEqual(s.tally.n_counted(POS_I1), 5)
        for cell in (POS_I2, NEG_I1, NEG_I2):
            run_block(r, cell, [1] * 5)
        self.assertTrue(s.tally.complete)
        for t in REAL_TYPES + PRACTICE_TYPES:
            s.select_type(t)
            r.frames("neutral", 15)
            self.assertEqual(s.start_block(r.t, override=True)[0], "complete")
            self.assertEqual(r.space().action, "refused")
        self.assertEqual(s.tally.n_counted_total, 20)
        self.assertIsNone(s.tally.suggested_type())

    def test_unmapped_class_fails(self):
        r = Rig(classes=["neutral", "bottle", "bowl", "typing"])
        try:
            for t in (NEG_I1, NEG_I2):
                res = r.trial(t, [("neutral", 5), ("typing", 20), ("neutral", 5)])
                self.assertEqual(res.verdict, FAIL)
                self.assertIn("Unmapped(typing)", res.set_text)
        finally:
            r.close()

    def test_invalid_does_not_use_slot(self):
        r, s = self.r, self.r.s
        run_block(r, NEG_I2, [1] * 4)
        s.select_type(NEG_I2)
        r.frames("neutral", 15)
        r.space()
        r.t += 0.4
        self.assertEqual(r.space().result.verdict, INVALID)
        self.assertEqual(s.tally.n_counted(NEG_I2), 4)
        self.assertEqual(s.tally.invalid_count, 1)
        run_block(r, NEG_I2, [0])
        self.assertEqual((s.tally.n_counted(NEG_I2), s.tally.successes(NEG_I2)), (5, 4))


# =========================================================================== void
class TestVoid(unittest.TestCase):
    def setUp(self):
        self.r = Rig()
        for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
            self.r.trial(p, seq)

    def tearDown(self):
        self.r.close()

    def test_void_rules(self):
        r, s = self.r, self.r.s
        run_block(r, POS_I1, [1, 0])
        self.assertEqual(s.void_last("", r.t).action, "refused")
        self.assertEqual(s.void_last("   ", r.t).action, "refused")
        ev = s.void_last("experimenter pressed SPACE early", r.t)
        self.assertEqual(ev.action, "voided")
        self.assertEqual(s.state, READY)
        self.assertEqual((s.tally.n_counted(POS_I1), s.tally.successes(POS_I1)), (1, 1))
        self.assertEqual(s.tally.voided_count, 1)
        ev = s.void_last("again", r.t)                    # only the most recent, once
        self.assertEqual(ev.action, "refused")
        self.assertEqual(s.tally.n_counted(POS_I1), 1)
        run_block(r, POS_I1, [1])
        self.assertEqual(s.tally.n_counted(POS_I1), 2)

    def test_void_frees_full_cell(self):
        r, s = self.r, self.r.s
        run_block(r, POS_I2, [1] * 5)
        s.void_last("wrong instruction given", r.t)
        self.assertFalse(s.tally.full(POS_I2))
        run_block(r, POS_I2, [0])
        self.assertEqual((s.tally.n_counted(POS_I2), s.tally.successes(POS_I2)), (5, 4))

    def test_not_while_recording_or_across_participants(self):
        r, s = self.r, self.r.s
        run_block(r, POS_I1, [1])
        s.select_type(POS_I1)
        r.frames("neutral", 15)
        r.space()
        self.assertEqual(s.void_last("x", r.t).action, "refused")
        r.frames("bottle", 15)
        r.space()
        self.assertTrue(s.next_participant("p2", r.t, force=True).ok)
        self.assertEqual(s.void_last("x", r.t).action, "refused")
        self.assertEqual(s.history[-1].n_counted(POS_I1), 2)

    def test_void_logged(self):
        r, s = self.r, self.r.s
        res = run_block(r, NEG_I1, [0])[0]
        s.void_last("participant sneezed on the bottle", r.t)
        rows = r.rows()
        trial_row = [x for x in rows if x["event"] == "trial" and x["trial_no"] == str(res.trial_no)][0]
        void_row = rows[-1]
        self.assertEqual(trial_row["voided"], "0")         # append-only: the trial row is not rewritten
        self.assertEqual((void_row["event"], void_row["trial_no"], void_row["voided"]),
                         ("void", str(res.trial_no), "1"))
        self.assertEqual(void_row["void_reason"], "participant sneezed on the bottle")
        self.assertEqual(void_row["verdict"], FAIL)


# =========================================================================== participants
class TestParticipants(unittest.TestCase):
    def setUp(self):
        self.r = Rig()

    def tearDown(self):
        self.r.close()

    def test_rules(self):
        r, s = self.r, self.r.s
        self.assertEqual(s.next_participant("  ", r.t).action, "refused")
        self.assertEqual(s.next_participant("P1", r.t).action, "refused")     # duplicate (case-insensitive)
        r.trial(PRACTICE_NEUTRAL, QUIET)
        self.assertTrue(s.next_participant("p2", r.t).ok)                     # practice only: no confirm
        run_block(r, POS_I1, [1])
        ev = s.next_participant("p3", r.t)
        self.assertEqual(ev.action, "refused")
        self.assertIn("1/20", ev.message)
        self.assertTrue(s.next_participant("p3", r.t, force=True).ok)
        self.assertEqual(s.tally.participant, "p3")
        self.assertEqual(s.selected_type, PRACTICE_NEUTRAL)
        self.assertEqual(s.tally.trials, [])
        self.assertEqual([t.participant for t in s.history], ["p1", "p2"])

    def test_not_while_recording(self):
        r, s = self.r, self.r.s
        r.frames("neutral", 15)
        r.space()
        self.assertEqual(s.next_participant("p9", r.t, force=True).action, "refused")

    def test_complete_then_next_writes_summary(self):
        r, s = self.r, self.r.s
        for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
            r.trial(p, seq)
        run_block(r, POS_I1, [1, 1, 1, 0, 0])
        run_block(r, POS_I2, [1, 1, 1, 1, 0])
        run_block(r, NEG_I1, [1, 1, 1, 1, 1])
        run_block(r, NEG_I2, [1, 1, 1, 1, 0])
        self.assertTrue(s.next_participant("p2", r.t).ok)                     # complete: no confirm
        summ = [x for x in r.rows() if x["event"] == "summary"][-1]
        self.assertEqual([summ[k] for k in ("participant", "pos_i1", "pos_i2", "pos_acc",
                                            "neg_i1", "neg_i2", "neg_acc", "counted")],
                         ["p1", "3", "4", "0.7", "5", "4", "0.9", "20"])


# =========================================================================== CSV
class TestCsv(unittest.TestCase):
    def test_rle(self):
        seq = ["neutral"] * 40 + ["bowl"] * 12 + ["neutral"] * 30
        self.assertEqual(rle(seq), '[["neutral",40],["bowl",12],["neutral",30]]')
        for s in ([], ["a*b", "a*b", "c > d", 'q"u,o', "é"], ["<press Calibrate Quiet>"]):
            self.assertEqual(unrle(rle(s)), s)
        self.assertEqual(unrle(""), [])

    def test_trial_row(self):
        r = Rig()
        try:
            res = r.trial(PRACTICE_I2, TAP_BOWL)
            rows = r.rows()
            self.assertEqual(list(rows[0].keys()), CSV_FIELDS)
            row = [x for x in rows if x["event"] == "trial"][0]
            expect = {
                "session_id": "S1", "participant": "p1", "trial_no": "1", "trial_type": PRACTICE_I2,
                "practice": "1", "label_set": "{Neutral, Interaction#2}", "verdict": PASS,
                "counted": "0", "voided": "0", "frames": "33", "error_frames": "0",
                "override_neutral": "0", "gate_db": "12.5", "frames_to_switch": "5", "sureness": "0.5",
                "neutral_class": "neutral", "i1_class": "bottle", "i2_class": "bowl",
                "timestamp": "2026-10-22T14:00:00.000",
            }
            for k, v in expect.items():
                self.assertEqual(row[k], v, k)
            self.assertEqual(unrle(row["shown_rle"]), res.shown_seq)
            self.assertEqual(float(row["duration_s"]), round(res.duration_s, 3))
            self.assertEqual(row["raw_rle"], "")
        finally:
            r.close()

    def test_raw_sequence_logged(self):
        r = Rig()
        try:
            r.frames("neutral", 15)
            r.space()
            for raw in ["neutral", "bowl", "bowl", "neutral"] * 3:
                r.frames("neutral", 1, raw=raw)
            res = r.space().result
            self.assertEqual(len(res.raw_seq), 12)
            self.assertEqual(unrle(r.rows()[-1]["raw_rle"]), res.raw_seq)
        finally:
            r.close()

    def test_rebuild_tallies_from_csv(self):
        r = Rig()
        try:
            for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
                r.trial(p, seq)
            run_block(r, POS_I1, [1, 0, 1])
            r.s.void_last("accidental start", r.t)
            run_block(r, NEG_I2, [1, 0])
            r.s.next_participant("p2", r.t, force=True)
            run_block(r, POS_I2, [1, 1])
            r.s.void_last("wrong instruction", r.t)
            rebuilt = tallies_from_csv(r.log.path)
            for tal in r.s.history + [r.s.tally]:
                for cell in REAL_TYPES:
                    self.assertEqual(rebuilt[tal.participant][cell],
                                     [tal.n_counted(cell), tal.successes(cell)], (tal.participant, cell))
        finally:
            r.close()

    def test_deferred_log_waits_for_first_trial(self):
        with tempfile.TemporaryDirectory() as d:
            log = TrialLog.for_session(d, "S9", defer_until_trial=True)
            s = TrialSession(RoleMap.from_classes(CLASSES), log=log, settings=SETTINGS, wall_clock=fixed_clock)
            t = 100.0
            self.assertTrue(s.next_participant("p1", t).ok)
            self.assertFalse(log.path.exists())  # participant row only: nothing written yet
            for _ in range(30):
                t += DT
                s.on_frame("neutral", t)
            self.assertEqual(s.space_down(t).action, "started")
            s.space_up(t)
            for _ in range(10):
                t += DT
                s.on_frame("neutral", t)
            self.assertEqual(s.space_down(t + 0.3).action, "stopped")
            with open(log.path, newline="", encoding="utf-8") as f:
                events = [row["event"] for row in csv.DictReader(f)]
            self.assertEqual(events, ["participant", "trial"])  # the held row is written first

    def test_locked_file_does_not_stop_trials(self):
        r = Rig()
        real_open = open

        def locked(*a, **k):
            raise PermissionError("file is open in another program")
        try:
            trial.open = locked
            res = r.trial(PRACTICE_NEUTRAL, QUIET)
            self.assertEqual(res.verdict, SET_ONLY)
            st = r.s.status(r.t)
            self.assertEqual(st["log_pending"], 1)
            self.assertIn("cannot write", st["log_error"])
            del trial.open
            r.trial(PRACTICE_I1, TAP_BOTTLE)
            self.assertEqual(r.log.pending, [])
            trial_rows = [x for x in r.rows() if x["event"] == "trial"]
            self.assertEqual([x["trial_no"] for x in trial_rows], ["1", "2"])
        finally:
            if "open" in vars(trial):
                del trial.open
            r.close()
        self.assertIs(open, real_open)

    def test_works_without_log(self):
        r = Rig(log=False)
        try:
            self.assertEqual(r.trial(PRACTICE_NEUTRAL, QUIET).verdict, SET_ONLY)
            self.assertIsNone(r.s.status(r.t)["log_error"])
        finally:
            r.close()


# =========================================================================== end to end
class TestEndToEnd(unittest.TestCase):
    def test_full_participant_and_status(self):
        r = Rig()
        try:
            s = r.s
            st = s.status(r.t)
            self.assertEqual((st["participant"], st["state"], st["suggested_type"]), ("p1", READY, PRACTICE_NEUTRAL))
            for p, seq in ((PRACTICE_NEUTRAL, QUIET), (PRACTICE_I1, TAP_BOTTLE), (PRACTICE_I2, TAP_BOWL)):
                r.trial(p, seq)
            # two taps in one window, a light tap missed, the other object struck by mistake
            r.trial(POS_I1, [("neutral", 4), ("bottle", 20), ("neutral", 10), ("bottle", 20), ("neutral", 5)])
            run_block(r, POS_I1, [1, 1, 0, 1])
            run_block(r, POS_I2, [1, 1, 1, 1, 1])
            run_block(r, NEG_I1, [1, 0, 1, 1, 1])
            run_block(r, NEG_I2, [1, 1, 1, 1, 1])
            st = s.status(r.t)
            self.assertTrue(st["complete"])
            self.assertEqual(st["cells"], {POS_I1: (5, 4), POS_I2: (5, 5), NEG_I1: (5, 4), NEG_I2: (5, 5)})
            self.assertEqual(st["start_block"][0], "complete")
            self.assertEqual(s.tally.sheet_rows()[0][3], 0.9)
            self.assertEqual(s.tally.sheet_rows()[1][3], 0.9)
            self.assertEqual(len([x for x in r.rows() if x["event"] == "trial"]), 23)
        finally:
            r.close()

    def test_thread_safety_smoke(self):
        r = Rig()
        try:
            s = r.s
            stop = threading.Event()
            errors = []
            clock = [r.t]

            def feeder():
                try:
                    while not stop.is_set():
                        clock[0] += DT
                        s.on_frame("neutral", clock[0])
                except Exception as e:  # pragma: no cover
                    errors.append(e)
            th = threading.Thread(target=feeder)
            th.start()
            for _ in range(200):
                s.press(clock[0] + 0.0001, override=True)
                s.status(clock[0])
            stop.set()
            th.join()
            self.assertEqual(errors, [])
            self.assertIn(s.state, (RECORDING, RESULT, READY))
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
