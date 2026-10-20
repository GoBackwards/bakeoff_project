# Presenter notes: what we adjusted and why

These notes explain how our vibration classifier works and what we changed from the course starter app, so the presenter can explain it during the demo and answer questions.

## The idea in one sentence

Everyday objects become buttons: **tapping the steel water bottle with a spoon is Interaction 1, tapping the ceramic bowl is Interaction 2**, and anything else (silence, talking, typing, handling things) is **Neutral**. The laptop recognizes which object is vibrating from the sound of its ring.

## How it works (30-second version)

1. **Sensing.** The laptop's built-in microphone picks up the object's vibration (its ring) after a tap.
2. **Spectrum.** About 44 times per second, the app takes the latest 50 ms of sound and computes its frequency spectrum (1103 frequency bins at 44.1 kHz).
3. **Ring gate.** Before classifying, the app asks: do a few frequencies stick out clearly above this room's normal noise? If not, the frame is **Neutral** right away. Silence, steady fan noise and a room that simply gets louder stay below the gate; rings, and also voices and clicks, pass it.
4. **Classifier.** Frames that pass the gate go to a linear support vector machine (scikit-learn `LinearSVC`) trained on our recordings of Neutral, bottle and bowl. It looks at the *shape* of the spectrum rather than its overall loudness, so a medium and a hard tap on the same object look alike.
5. **Steadying the answer.** A frame counts as a tap only if the model is reasonably sure, and a tap is shown only after 5 such frames in a row (about 0.11 s). It then stays on screen for at least 0.4 s. Single-frame mistakes never reach the screen.

## What we adjusted from the starter app

| Problem we hit | What we changed | Result |
|---|---|---|
| In silence, the prediction kept jumping between all three classes | **Calibrate Quiet button**: 3 s of the quiet room teach the app what "nothing happening" sounds like at each frequency, and set a gate a little above it. Quieter frames are Neutral and never classified. Quiet gaps between taps are left out of the bottle and bowl training data, even for recordings made before calibrating (retrain after calibrating) | In simulation no quiet frame passed the gate. On our real 3 s quiet recording the gate stopped about 96% of frames, and with 5 frames to switch the displayed label stayed Neutral the whole time |
| The gate missed soft taps, or opened when the laptop fan got louder | The gate measures how far the **2 strongest frequencies stand out from the typical frequency**, compared with the room's own noise, instead of overall loudness. Tuned on a simulation of the app's audio pipeline | In simulation, about 18 dB more sensitive to a bottle-like ring than our previous 10-bin gate, with no false openings in a quiet room, a room 12 dB louder, or with a faint new fan tone (clearly audible new tones still reach the classifier) |
| One cough during calibration could ruin the gate | Calibration uses the room's *typical* level (the median), not its loudest moment, and warns if the room was noisy | Mixing our loudest recorded sounds into a calibration moved the gate by less than 0.4 dB |
| Single-frame flickers, key clicks | A tap must last **5 frames in a row** before it is shown, then stays up at least 0.4 s | Short blips and keyboard clicks are hidden |
| Saying "wow", typing or clicking the mouse sometimes showed a tap | **5 frames to switch plus a sureness setting**: a frame counts as Neutral unless the model is clearly sure it is a tap (top score at least 0.5 above the runner-up). We also record talking, typing and clicking as Neutral examples | On held-out Neutral recordings (about 1.5 min): about 17 false taps per minute with our earlier settings (3 frames, no sureness), about 3 per minute with ours. Most of the gain comes from the 5-frame rule (alone: about 5 per minute); sureness removes about a third of what is left |
| A soft tap and a hard tap on the same object looked different to the model | Features are log-scaled with the frame's average removed (spectral shape only), and the scaling is saved inside the model so training and live use always match | Overall loudness no longer matters: medium and hard taps read the same (94-99% of frames correct). The softest taps are still less reliable (about 60-67% of frames), which is why we tap firmly |
| The app froze after stopping one recording and starting another | The spectrum is now updated from a steady loop instead of a chain of "call me next frame" requests (which broke for good when one request came late), and saving runs in the background | 25 s of repeated record / stop / switch cycles with no pause longer than 0.03 s |
| Recordings were lost whenever the app closed | Recordings and the calibration are saved automatically, safely (a crash mid-save cannot corrupt them), and reloaded at start. The trained model is not reloaded: train again or use Load Model | Data survives restarts |
| Starter bugs | Fixed the swapped Clear All / Delete All buttons, the Train button that could stay disabled, errors that froze the app, microphone switching, and duplicate frames being recorded | — |

We also wrote a hardware go/no-go test (`tools/gonogo.py`, about an hour of recording) for checking that a pair of objects can be told apart from each other and from everyday sounds before committing to them. It was written while we were considering a steel bowl and a glass jar.

## Numbers you can quote

All from our own recordings and simulations, not from the live bakeoff:

- **Per-frame accuracy on held-out 2-second stretches of our recordings:** about 90% for Neutral, 85% for the bottle, 83% for the bowl. The 5-frame rule and the sureness setting exist because single frames are not perfect; whole taps are much more reliable.
- **False taps on everyday sounds** (talking, handling, typing): about 17 per minute with our earlier 3-frame settings, about 3 per minute with ours.
- **Ring frequencies we measured** (as read off the app's spectrum): the bottle rings mainly around 360, 740 and 1360 Hz; the bowl around 640, 1280 and 3100 Hz.
- **Speed:** about 44 frames per second; a tap appears roughly 0.15-0.25 s after it starts.

## Before the demo (checklist)

1. Start the app: `uv run visualizer.py`.
2. Keep the laptop plugged in; turn off microphone noise cancellation (Armoury Crate "AI Noise Cancelation", Windows audio enhancements).
3. In the demo room, press **Calibrate Quiet** and keep everyone quiet for the next 3 s, until the status line shows the gate (about 11-13 dB in our tests; much higher values discard soft taps). Our laptop currently has a hand-raised gate of 24.8 dB saved, so always recalibrate. Recalibrate again if the room changes.
4. In Train mode, click **Train Model** (a few seconds; it switches to Infer when done), or use **Load Model** for a saved `.model` file. The model is not reloaded automatically.
5. In Infer mode, set **frames to switch = 5** (our laptop currently has 3 saved) and **sureness = 0.50** (type it and press Enter).
6. Quick self-test: silence stays Neutral; talking and typing stay Neutral; a firm tap on each object shows the right label.
7. Tap **firmly** with the spoon. Very light taps, whose ring does not stand out roughly 15-20 dB above the room for at least a tenth of a second, are treated as Neutral by design.

## Likely questions

**Why not a contact sensor?** We used the laptop's built-in microphone so the setup needs no extra hardware and stays portable. The microphone picks up each object's ring, which is the object vibrating at its own natural frequencies.

**How does it tell the two objects apart?** Each object rings at its own set of frequencies (its resonant modes), which depend on its material and shape (and, for the bottle, how much water is in it), not on how hard you tap. The classifier learns those spectral patterns.

**Why does talking not trigger it?** Talking usually passes the gate, because it sticks out above a quiet room. What stops it is that the classifier was trained on lots of talking as Neutral, uncertain frames count as Neutral, and a tap must last 5 frames. The bottle's ring frequencies overlap the voice range, so recording plenty of talking as Neutral matters.

**What happens in a noisy room?** The gate is calibrated to the room's own noise, and a room that just gets louder does not open it. New noises that stick out (a nearby tone, a loud voice) reach the classifier, which is why we train Neutral on many everyday sounds.

**What would you improve with more time?** A detector that looks at a whole tap at once (onset plus ring) instead of single frames, and more Neutral data from more people.

## Where things are in the code

| What | Where |
|---|---|
| Gate, calibration maths, smoothing, sureness, features | `models.py`: `frame_level_db`, `calibrate_noise`, `LabelSmoother`, `predict_confident`, `preprocess_data`, `train_model` |
| App, buttons, recording, saving, main loop | `visualizer.py` |
| Hardware go/no-go test | `tools/gonogo.py` |
