# Clink

**Ask for help from bed by tapping what's already on the nightstand.**

Tap the water bottle: "Water, please." Tap the bowl: "Please come." No buttons, no batteries, nothing attached to the objects.

This is the team's story for the vibration bakeoff: the problem, who it is for, why tapping objects, what each tap means, and how to present and run the demo. For how the classifier works and what we changed from the starter, see [PRESENTER_NOTES.md](PRESENTER_NOTES.md).

## The problem

Someone resting in bed or in a chair at home often needs small, non-urgent help, for example while recovering from an operation or an illness. They might want a water refill, help sitting up, help getting to the bathroom, or company. The person who can help is usually in another room. Today the options are poor:

- **Shout.** This fails if your voice is quiet or the door is shut, and many people find it undignified.
- **Phone or text.** The phone is often out of reach, locked or flat, and it is hard to use lying down.
- **Bang on something or ring a bell.** That gets attention but carries only one meaning. The helper has to walk in, ask "what do you need?", walk out and come back. A bell also reaches only as far as someone can hear it.
- **Wait.**

Clink turns two ordinary objects on the nightstand into a call system with two meanings. A laptop (later a phone or an old tablet) hears which object was struck and shows the request. It is meant for frequent, everyday requests. **It is not an emergency alarm.**

## Who it's for

- **Primary user:** a person who spends long stretches in bed or a chair at home and finds calling out or using a phone hard, but who can still hold a spoon and give one firm tap. Their voice may be quiet, the phone may be out of reach, or they may not want to shout across the house.
- **Secondary user:** the family member or housemate in the kitchen or at a desk who wants to know "do they need me, and what for?" without hovering.
- **In the demo:** "my grandmother in the bedroom" and "me in the kitchen". These are personas. We did not study real users.
- **Honest scope:**
  - The firm tap excludes people with severe weakness or strong tremor.
  - Today's prototype is a laptop within arm's reach of the objects, which is also the nightstand setup of the vision.
  - Sending the request to the helper's phone is **not built**.

## Why tapping objects (vs phone, voice, smart devices)

1. **The object is the label.** You tap the thing you want. There is nothing to memorise, no menu and no reading. A visitor or a second carer learns it from one demonstration, in any language.
2. **A coarse, eyes-free movement.** You can find the bottle by touch, lying down, without glasses. A tap uses the whole arm, and there is no small target behind a lock screen.
3. **It confirms itself.** You feel the spoon hit and hear the ding before any screen reacts.
4. **Nothing on the objects.** There is nothing to charge, pair, mount or keep dry, and you keep your own bottle. The only device is one the household already owns.
5. **Compared with the alternatives:**
   - *Phone:* no unlocking, no app, no reading.
   - *Voice assistant:* no need for a loud, clear voice or a wake word, and we never turn sound into words. Taps complement voice rather than replace it.
   - *Wireless call button:* good for one meaning, and probably more reliable than our prototype today. Clink carries two meanings without adding any hardware.
   - *Hand bell:* one meaning. A listening device can tell **which** object rang.

**Why vibration:**
- A struck object rings at its own resonant frequencies, which depend on its material and shape.
- In our recordings the bottle rings mainly around 360, 740 and 1360 Hz, and the bowl around 640, 1280 and 3100 Hz.
- The app recognises each object's vibration fingerprint with overall loudness removed. It does not recognise the person or their words.
- The objects are passive resonators. The laptop microphone picks up the sound their vibration sends through the air.
- A contact sensor under the nightstand is the natural next version.

## What the two taps do (and why that mapping)

**Interaction #1: a spoon tap on the steel water bottle means "Water, please".**
- The most common specific request lives on the object it is about. You tap the thing you want refilled, so there is no mapping to learn.
- It saves the helper a round trip, because they arrive with water the first time.
- The bottle is the object most likely to be on a nightstand already.

**Interaction #2: a spoon tap on the ceramic bowl means "Please come".**
- This is the open, general call: sitting up, the bathroom, the remote, company.
- It builds on familiar conventions. Tapping a glass or dish with a spoon gets a room's attention, and a struck bowl rings like a small hand bell, the traditional sickroom call.
- It is a dedicated "bell bowl" that stays on the nightstand, not the bowl you eat from, so meals don't ring it.

**Why bottle = 1 and bowl = 2.** The bottle carries the one specific, frequent request, and the bowl carries everything else. One specific request plus one open call covers most small needs with two gestures. The two must never be confused: a "Please come" read as "Water, please" sends the wrong help.

## Why Neutral matters

Everyday life goes on next to the listener all day: visitors talking, phone calls, the TV, picking the bottle up and putting it down, shifting in bed, someone typing. If any of that pages the helper, each false alarm pulls them away for nothing. After a few, they stop trusting the system, and then a real request gets ignored too (alarm fatigue).

A missed tap shows itself: no request appears, so you tap again. A false alarm gives no such sign, and the helper pays for it. **Neutral is the product's most important behaviour, not a leftover class.**

The bakeoff scores it the same way:
- Half the trials are Negative trials, and one wrong label anywhere in the window fails the trial.
- Tapping the *other* object is a legal negative, so a bottle/bowl mix-up fails too.

How the app puts Neutral first:
- **Calibrate Quiet:** 3 s of the quiet room set a ring gate. A frame whose strongest frequencies don't stand out above the room's own noise is Neutral and is never classified.
- **Neutral training data:** we recorded talking, typing, clicking and handling as Neutral.
- **Sureness:** a frame counts as a tap only if the top score beats the runner-up by at least 0.5.
- **5 frames to switch:** a tap is shown only after 5 agreeing frames (about 0.11 s of ring), so one-frame blips never reach the screen. A shown tap is held for at least 0.4 s.
- **Light taps are ignored by design.**

**Honest status:** on our own held-out recordings of continuous talking, typing and handling, about 3 false taps per minute remain (down from about 17 with the earlier starter-style settings). That is acceptable for trial windows of a few seconds, but not for an all-day pager.

## What the app shows

`uv run visualizer.py` opens one window with a calm light look (linen page, white cards). The top bar shows the name, the tagline and the SETUP / TRAIN / DEMO switch, plus a red note only when something is missing (no model, no microphone). Two things show on every screen:
- **NOW HEARING card:** the live label in large type with its request. It is white with grey "Neutral - Listening", and the whole card turns solid sky blue for "Bottle - Water, please" or amber for "Bowl - Please come".
- **Ring level meter:** a bar with a dark tick at the gate. It twitches when you talk and jumps past the tick when an object rings. On SETUP and TRAIN a small "raw:" note shows the model's guess before smoothing.

**SETUP:**
- Microphone choice and **Calibrate Quiet**, with a progress bar and the gate in dB.
- Recognition settings: frames to switch and sureness.
- **Bakeoff roles:** which trained class is Interaction#1 (default: the class named `bottle`) and which is Interaction#2 (default `bowl`).
- A "Ready for the demo?" checklist, plus the full spectrum with the bottle's and bowl's ring frequencies marked.

**TRAIN:** the class table (frames per class, Select / Clear / Delete), the big Record button, and the Train / Save / Load model buttons. Every trained model is also kept in `metadata/last_trained.model` and loaded again at the next start.

**DEMO** (opens by itself once a model is live):
- **Left, the product view:**
  - A request headline: "All quiet", or for example "Water, please - 14:32" in the object's colour, held for 3 s so people can read it.
  - Two tiles: "Tap the bottle: Water, please" and "Tap the bowl: Please come". A tile flashes on its request, and its counter ("water requests", "calls") goes up by one per request. A request is counted once each time the shown label changes to that object, never once per frame.
  - A short history with times, and a small spectrum.
- **Right, the bakeoff trial recorder:** see the next section.

## Running a bakeoff trial (experimenter guide)

The recorder follows the slides. SPACE starts a trial. Every label shown on screen until the next SPACE goes into a set. SPACE again stops the trial and shows the set and the verdict:

| Trial | Passes only if the set is |
|---|---|
| Positive Bottle (Interaction#1) | {Bottle} or {Neutral, Bottle} |
| Positive Bowl (Interaction#2) | {Bowl} or {Neutral, Bowl} |
| Negative Bottle | {Neutral}, {Bowl} or {Neutral, Bowl} |
| Negative Bowl | {Neutral}, {Bottle} or {Neutral, Bottle} |

Before the participants arrive:
1. Start the app. In **SETUP**, press **Calibrate Quiet** with the room quiet for 3 s. The gate should read about 11-13 dB. A much higher gate makes soft taps disappear.
2. Set **frames to switch = 5** and **sureness = 0.50** (type the value and press Enter). Check that the checklist shows every box ticked.
3. Make sure a model is live: it loads by itself at start, or click **Train Model** / **Load Model** in TRAIN. The app then switches to **DEMO**.
4. Click into the DEMO screen once so the window has the keyboard. If a text box has the cursor, SPACE types into it instead, so press Enter or click elsewhere first.

For each participant:
1. The app starts with participant **P1**. For the next person, type the ID in the box next to **Next participant** (it suggests P2, P3...) and click the button.
2. **Practice:** the recorder selects Practice Neutral, then Practice Bottle, then Practice Bowl. For each one, read the line in the box under the banner aloud. You may explain and help. Press **SPACE**, let them act, then press **SPACE** again. Practice is shown but never counted, and each practice can be run once. Practice Neutral has no verdict: the banner shows DONE with the set, for example `DONE {Neutral}`.
3. **Real trials:** the recorder then selects Positive Bottle, Positive Bowl, Negative Bottle and Negative Bowl in turn, 5 of each. You can click another trial type at any time. For each trial:
   - Read only the line in the box, for example "Tap the bottle." or "Anything but the bottle.", and give no help. The small caption above it says whether you may help.
   - Press **SPACE** and say "go".
   - After a tap, wait about half a second before pressing **SPACE** again. The label appears about 0.15-0.25 s after the tap.
   - The banner shows **PASS** or **FAIL** with the set, for example `PASS {Neutral, Bottle}`, and the tally table updates.
4. Copy the result into the projected spreadsheet. **Copy numbers** puts the two rows on the clipboard as tab-separated cells (Interaction#1, Interaction#2, Accuracy), ready to paste.

What the recorder guards against:
- **WAIT FOR NEUTRAL:** a trial can start only after the screen has shown Neutral for 0.3 s, so a tap still on screen never leaks into the next trial. **Shift+SPACE** or the **Start anyway** button skips this, and the log marks that trial.
- **Esc** cancels a running trial. It is marked INVALID, is not counted, and can be run again. Leaving the DEMO screen or changing the microphone does the same.
- **INVALID** also covers an app problem during the window: no audio, a lost frame, a settings change, or "press Calibrate Quiet". An INVALID trial never passes and is not counted.
- A 6th trial of a type that already has 5 is refused, and so is a 21st trial overall (COMPLETE).
- **Void last trial** is for an experimenter mistake, such as the wrong type selected or SPACE pressed too early. It needs a reason, applies only to the most recent trial, and frees its slot. The trial stays in the log, marked void.
- Holding SPACE down acts once, and SPACE does nothing on the SETUP and TRAIN screens.

Every trial, void and participant change is written to `results/trials_<date>_<time>.csv`, one file per app run, created at the first trial. Each row records:
- the set and the verdict;
- the frame-by-frame label sequence;
- the gate, frames to switch and sureness used;
- the class roles.

The console also prints each set when a trial stops.

## 60-second demo script for the presenter

**Setup:**
- The app is on DEMO with the model live, frames to switch 5 and sureness 0.50.
- Calibrate Quiet has been done in this room with the room quiet, and noise cancellation is off.
- The bottle is at its training fill level, and both objects are where they were in rehearsal.
- The presenter holds the spoon.

**Pitch (about 150 words):**
> "Picture my grandmother resting in bed after an operation. I'm in the kitchen. Her voice is quiet and her phone is on the dresser, so a glass of water means shouting or waiting.
> Clink turns what's already on her nightstand into a call system. No buttons, no batteries, nothing on the objects.
> *[One firm tap on the bottle; the screen turns blue.]* Tap the water bottle: 'Water, please.' You tap the thing you want, so there's nothing to remember.
> *[One firm tap on the bowl; the screen turns amber.]* Tap the bowl, like a little bell: 'Please come.' In the product this would reach my phone. Today it shows here.
> *[Point at the spectrum.]* The laptop's microphone hears which object is vibrating. Each one rings at its own frequencies: the bottle here, the bowl here.
> *[Keep talking; type a few keys.]* Just as important: talking, typing and moving around stay Neutral, because every false alarm drags a helper away for nothing and teaches them to ignore it."

**Stage rules:**
- Add "lift the bottle and set it down gently" only if it never fired in that day's rehearsal.
- Never stir or scrape the bowl.
- Don't grab an object right after tapping it. Let it ring.

**Practice briefing** (say this to each visiting participant, about 25 s; it is the only time we may help):
> "You get one practice of each, then we can't help. Hold the spoon like a little hammer. Bottle: one firm tap on the side, at the dot, like ringing a bell, and let it ring. *[Point when Bottle appears.]* Bowl: the same, one firm tap on the side at the dot. *[Point.]* Neutral: for a few seconds, do anything except tap the bottle or the bowl. Talk, move your hands, touch the table. *[Point at Listening.]* Gentle taps are ignored on purpose, so tap like you mean it. In the real trials we'll only name the task. 'Tap the bottle' means one firm tap on the bottle. 'Anything but the bottle' means do whatever you like except tap the bottle, and tapping the bowl is allowed."

**Experimenter tips:**
- In Negative trials, let the participant act for a few seconds.
- Don't change the bottle's water level between participants.
- Recalibrate (with the room quiet) if the room's steady noise changes, for example when the projector or HVAC turns on.
- A small tape dot on each object shows where to tap. It is not electronics. Put it where we tapped while recording the training data.

## Hard questions and honest answers

**Q1. "Why not buy a cheap wireless call button?"**
For one meaning, a call button is a good answer and probably more reliable than our prototype today. Our claim is narrower:
- nothing is added to the objects;
- the objects carry two distinct meanings that need no labels;
- the listener can be a device the household already owns.

We haven't compared the two with real users.

**Q2. "Is a microphone really vibration sensing?"**
- The signal is the object's own structural vibration (its resonant modes), reaching the microphone through the air.
- The features are spectral shape with loudness removed, so the model identifies the resonator, not how hard it was hit.
- Because the path is through air, the mic also hears unrelated sounds, which is why Neutral training matters. A contact sensor would hear less speech.
- We chose the built-in mic so the setup needs no hardware. (Confirm with the TA that this fits the theme.)

**Q3. "Can the people you design for actually tap firmly?"**
Only some of them. Clink needs enough grip to hold a spoon and a roughly aimed, firm tap, and light taps are treated as Neutral by design. People with severe weakness or tremor may be excluded. A thick-handled adaptive spoon might help, but we haven't tested that, and we have not tested with patients, older adults or caregivers.

**Q4. "Real life isn't a 5-second trial. What about the TV, visitors, putting the bottle down?"**
This is our biggest open problem. About 3 false taps per minute remained on our held-out everyday sounds: fine for trials, far too many for an all-day pager. Planned fixes:
- a tap detector that looks at whole taps (sharp onset plus decaying ring) instead of 50 ms frames;
- an optional double tap before anything is sent;
- a dedicated bell bowl;
- a dismiss button on the helper's side.

We have not tested set-down clinks or the TV specifically.

**Q5. "The bottle's ring changes with its water level, and it's emptiest exactly when you want water."**
Correct, and a good catch. We trained at one fill level and keep the bottle there for the demo. A product would need training data at several fill levels and a quick per-household enrolment. We haven't measured how far the bottle's ring shifts between full and empty.

**Q6. "An always-on microphone in a bedroom?"**
A fair concern. The app computes 50 ms spectra in memory, classifies them and discards them. It saves no audio, and saves spectra only while you press Record in TRAIN (the trial log keeps only labels). Nothing is uploaded, and it never tries to recognise words. A product would still need explicit consent from the person in bed, a visible "listening" indicator and a physical mute. A contact sensor would hear much less speech.

**Q7. "Why not a voice assistant?"**
Voice works well when the person can speak clearly and loudly enough and is happy to. Clink targets the moments when they can't or won't: a quiet voice, a sleeping partner, no wish to say a wake word. It needs no speech recognition and works in any language. We see it as a complement to voice.

**Q8. "Hasn't tap sensing on objects been done?"**
Yes. There is research on acoustic and vibration sensing of taps and scratches on everyday surfaces and objects, and we don't claim a new sensing method. Our contribution is the interaction design (passive objects whose meaning is the object itself) and a Neutral-first pipeline that holds up for strangers under bakeoff rules.

**Q9. "Why only two requests? What about emergencies?"**
Two requests cover the common case simply. Two distinct inputs can also drive "step / select" scanning through a longer menu, as some two-switch access methods do. Clink is not an emergency alarm and must never replace one.

**Q10. "Does your trial recorder decide on the raw classifier output?"**
No. It collects the label shown on screen, after the gate, sureness and the 5-frame smoother, because that is what the product would act on. We say this openly, and the log keeps the raw per-frame guesses too.

## Limitations and what we'd build next

**Limitations:**
- **No user research.** The users, the situation and the benefits are hypotheses.
- **Not built:** the relay to the helper's phone.
- **False alarms:** about 3 false taps per minute on our held-out everyday sounds.
- **Firm taps only.** The softest taps were only about 60-67% of frames correct and are treated as Neutral by design, which demands grip and force.
- **One setup only:** trained on one bottle, one bowl, one laptop and one room.
  - The bottle's ring depends on its fill level.
  - Recalibrate in every room, and keep Windows / Armoury Crate noise cancellation off.
  - We haven't characterised the working distance.
- **Untested sounds:** set-down clinks, eating noise and the TV.
- **Privacy:** an always-on microphone in a bedroom.
- **Scope:** only two meanings, no Undo, not an emergency system.

**What we'd build next:**
1. **Relay to the helper's phone**, with a dismiss button and "Water, please x2" merging for repeats.
2. **A whole-tap detector** (onset plus ring) and an optional double tap, to cut false alarms.
3. **An "enrol your objects" step:** a minute of taps per object, at several fill levels.
4. **A contact vibration sensor** under the nightstand, for less speech and more privacy.
5. **Running on a phone or an old tablet** instead of a laptop.
6. **A small study** with care partners, with input from an occupational therapist.

## Claims to verify before presenting

**Our own numbers.** These come from [PRESENTER_NOTES.md](PRESENTER_NOTES.md). Say "in our recordings", and re-measure after any retraining:
- Ring frequencies: bottle about 360/740/1360 Hz, bowl about 640/1280/3100 Hz. Re-check them on the demo objects at the demo fill level.
- About 3 vs about 17 false taps per minute.
- Per-frame accuracy about 90/85/83% (Neutral/bottle/bowl), and about 60-67% for soft taps.
- A tap appears about 0.15-0.25 s after it starts. 5 frames is about 0.11 s of ring.

**Demo-day checks** (rehearse in the actual room):
- Does gripping an object right after a tap cut the ring below 5 frames?
- Does tap location matter? This decides where the tape dot goes.
- How far does the bottle's ring shift as its water level changes?
- Do lifting and setting down the bottle, or the presenter's voice during the pitch, ever fire?
- Does the classroom's projector or HVAC need a recalibration?
- Run one full mock participant (3 practice + 20 trials) with the trial recorder, and paste the numbers into a copy of the spreadsheet.

**Outside claims** (find a source or phrase them qualitatively):
1. People recovering at home often need small, non-urgent help, and the helper is often out of earshot.
2. Voice assistants are less reliable for quiet or atypical speech (for example dysarthria).
3. Alarm fatigue: frequent false alarms lead people to ignore alarms. Cite a review and don't invent figures.
4. Tapping a glass with a spoon to get attention, and the sickroom hand bell, are familiar conventions. These vary by culture.
5. Two-switch step/select scanning is an established access method in assistive technology.
6. Adaptive thick-handled cutlery exists.
7. Inexpensive wireless call buttons exist. Give no price without a source.
8. Resonant frequencies depend on material, geometry and (for a bottle) fill level. A textbook source is enough.
9. Related work, if novelty comes up: check authors, venues and years for Scratch Input, Acoustic Barcodes, Touch & Activate and Ubicoustics before naming them.

**Process** (confirm with the TA):
- Airborne pickup of an object's vibration counts as "vibration-based".
- The trial set may collect the system's smoothed output (what the screen shows) rather than raw per-frame labels.
- INVALID trials (an app problem, or Esc) are re-run rather than counted as failures, and voiding an experimenter mistake is allowed. If the TA disagrees, `TrialSession(invalid_counts_as_fail=True)` in `visualizer.py` counts INVALID as a failure instead.
