import numpy as np, wave
from scipy.signal import butter, sosfilt
SR = 44100; DUR = 50.0; N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(3)
tt = np.arange(N) / SR

def filt(x, kind, f):
    sos = butter(2, f, btype=kind, fs=SR, output='sos'); return sosfilt(sos, x)
def add(sig, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR); sig = sig[:max(0, N - i)]
    if i < 0: sig = sig[-i:]; i = 0
    L[i:i+len(sig)] += sig * gain * (1 - max(0, pan)); R[i:i+len(sig)] += sig * gain * (1 + min(0, pan))
def env(n, a, d):
    t = np.arange(n) / SR; return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)
def noise(sec): return rng.standard_normal(int(sec * SR))

# ---------- music: tempo follows speed ----------
def bpm_at(t):
    pts = [(0, 96), (4.5, 100), (10, 118), (13, 132), (16.6, 148), (20.6, 160), (26, 168), (31, 172), (38, 90), (42.6, 96), (44.6, 110), (50, 110)]
    for (a, x), (b, y) in zip(pts, pts[1:]):
        if a <= t < b: return x + (y - x) * (t - a) / (b - a)
    return 110
beats = []; t = 0.0
while t < DUR:
    beats.append(t); t += 60 / bpm_at(t) / 2  # eighth notes
kick_len = int(.35 * SR); kt = np.arange(kick_len) / SR
kick = np.sin(2 * np.pi * (45 * kt + 70 * (1 - np.exp(-kt * 30)) / 30 * 1)) * np.exp(-kt * 9)
kick = np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-kt * 35)) / SR) * np.exp(-kt * 8)
hat = filt(noise(.06), 'highpass', 7000) * env(int(.06 * SR), .001, .015)
snare = filt(noise(.2), 'bandpass', [1500, 6000]) * env(int(.2 * SR), .001, .05) + np.sin(2 * np.pi * 190 * np.arange(int(.2 * SR)) / SR) * env(int(.2 * SR), .001, .04) * .5
bassline = [55, 55, 65.4, 49]
for k, b in enumerate(beats):
    if 36 <= b < 38: continue  # silence during time reversal
    quiet = .45 if (38 <= b < 42.6) else 1
    if k % 4 == 0: add(kick, b, .55 * quiet)
    if k % 4 == 2 and b > 4.5: add(snare, b, .22 * quiet)
    add(hat, b + (0.5 * 60 / bpm_at(b) / 2 if k % 2 else 0), .12 * quiet, pan=.3 if k % 2 else -.3)
    if b > 4.5:
        f = bassline[(k // 8) % 4] * (2 if b > 20.6 else 1)
        n = int(60 / bpm_at(b) / 2 * SR); x = np.arange(n) / SR
        add(np.sign(np.sin(2 * np.pi * f * x)) * .5 * np.exp(-x * 6) + np.sin(2 * np.pi * f * x) * np.exp(-x * 4), b, .09 * quiet)

# ---------- footsteps ----------
for t0 in np.arange(4.7, 10, .42 - 0):
    s = filt(noise(.08), 'lowpass', 900) * env(int(.08 * SR), .001, .02); add(s, t0, .35)

# ---------- wind rising with speed ----------
w = filt(noise(DUR), 'bandpass', [300, 3000])
wg = np.interp(tt, [0, 8, 13, 16.6, 20.6, 26, 26.01, 31, 31.01, 36, 38, 38.01, 42.6, 50], [0, 0, .08, .16, .22, .25, 0, 0, .2, .35, 0, .03, .03, 0])
L += w * wg; R += np.roll(w, 300) * wg

# ---------- electric crackle ----------
for t0 in np.concatenate([np.arange(11.4, 26, .045), np.arange(31, 36, .04), np.arange(38, 42.6, .09), np.arange(0.95, 1.6, .04), np.arange(43.0, 43.7, .04)]):
    if rng.random() < .55:
        s = filt(noise(.03), 'highpass', 3000) * env(int(.03 * SR), .0005, .006); add(s, t0 + rng.random() * .03, .35, pan=rng.uniform(-.6, .6))
def zap(t0, g=.5):
    n = int(.5 * SR); x = np.arange(n) / SR
    s = (filt(noise(.5), 'highpass', 1500) * (np.sin(2 * np.pi * 60 * x) > 0) + np.sin(2 * np.pi * np.cumsum(2000 * np.exp(-x * 6) + 200) / SR) * .4) * env(n, .002, .12)
    add(s, t0, g)
for t0 in [11.5, 13, 15, 16.6, 19, 20.6, 22, 23.5, 26, 27.2, 28.4, 29.6]: zap(t0, .35)

# ---------- whooshes ----------
def whoosh(t0, dur=.7, g=.6, pan_from=-1, pan_to=1, lo=400, hi=4000):
    n = int(dur * SR); x = np.linspace(0, 1, n)
    s = filt(noise(dur), 'bandpass', [lo, hi]) * np.sin(np.pi * x) ** 2
    p = pan_from + (pan_to - pan_from) * x
    i = int((t0 - dur / 2) * SR); j = min(N, i + n)
    L[i:j] += (s * g * np.clip(1 - p, 0, 1))[:j - i]; R[i:j] += (s * g * np.clip(1 + p, 0, 1))[:j - i]
whoosh(1.25, .55, 1.0); whoosh(43.25, .55, 1.0)
for t0 in [20.6, 22.6, 26, 31, 44.6]: whoosh(t0, .8, .5, 1, -1)
whoosh(38.2, 1.4, .5, -1, 1, 200, 2000)

# ---------- cartoon accents ----------
def boing(t0, f0=300, g=.3):
    n = int(.45 * SR); x = np.arange(n) / SR
    add(np.sin(2 * np.pi * np.cumsum(f0 + 180 * np.sin(x * 40) * np.exp(-x * 5)) / SR) * env(n, .005, .15), t0, g)
boing(1.55); boing(41.1, 340); boing(43.55, 300, .2)
def pop(t0, g=.35):
    n = int(.12 * SR); x = np.arange(n) / SR
    add(np.sin(2 * np.pi * np.cumsum(900 - 600 * x / .12) / SR) * env(n, .001, .03), t0, g)
for t0 in [11.5, 20.6, 23.5, 36.1]: pop(t0)

# ---------- sonic boom ----------
n = int(2.5 * SR); x = np.arange(n) / SR
boom = filt(noise(2.5), 'lowpass', 400) * env(n, .003, .5) * 1.6 + np.sin(2 * np.pi * np.cumsum(30 + 60 * np.exp(-x * 8)) / SR) * env(n, .002, .6) * 1.2
crack = filt(noise(.15), 'highpass', 1500) * env(int(.15 * SR), .0005, .03)
add(boom, 16.7, .9); add(crack, 16.7, .8)
# glass tinkles
for k in range(70):
    t0 = 16.8 + rng.random() * 2.6; f = rng.uniform(3000, 7000); m = int(.15 * SR); y = np.arange(m) / SR
    add(np.sin(2 * np.pi * f * y) * env(m, .0005, .03), t0, .07, pan=rng.uniform(-.8, .8))

# ---------- explosion ----------
n = int(3 * SR); x = np.arange(n) / SR
ex = filt(noise(3), 'lowpass', 700) * env(n, .005, .7) * 1.4 + filt(noise(3), 'lowpass', 120) * env(n, .01, 1.2) * 1.5
add(ex, 19.6, .85)
# fire roar 19.2-26
fr = filt(noise(7), 'bandpass', [80, 900]) * np.clip(np.linspace(0, 4, int(7 * SR)), 0, 1)
fr *= np.clip(np.linspace(7, 0, int(7 * SR)) * 2, 0, 1)
add(fr, 19.2, .18)

# ---------- water ----------
wa = filt(noise(3.4), 'bandpass', [600, 6000]) * (.6 + .4 * np.sin(np.arange(int(3.4 * SR)) / SR * 40))
wa *= np.clip(np.linspace(0, 6, len(wa)), 0, 1) * np.clip(np.linspace(6, 0, len(wa)), 0, 1)
add(wa, 22.6, .22)


# ---------- water explosions ----------
for t0, sc in [(22.85, 1), (23.5, 1.6), (24.15, 1), (24.8, 1.2), (25.45, 1.1)]:
    n = int(2 * SR); x = np.arange(n) / SR
    sp = filt(noise(2), 'bandpass', [400, 7000]) * env(n, .004, .35) + filt(noise(2), 'lowpass', 150) * env(n, .005, .25) * 1.5
    add(sp, t0, .5 * sc, pan=rng.uniform(-.3, .3))

# ---------- warp riser + time reversal ----------
n = int(5 * SR); x = np.arange(n) / SR
riser = np.sin(2 * np.pi * np.cumsum(120 * 2 ** (x * 1.1)) / SR) * (x / 5) ** 1.5
riser += filt(noise(5), 'highpass', 2000) * (x / 5) ** 2 * .4
add(riser, 31, .3)
n = int(2 * SR); x = np.arange(n) / SR
rev = np.sin(2 * np.pi * np.cumsum(2000 * 2 ** (-x * 3)) / SR) * np.sin(np.pi * x / 2) ** 2
tick = np.zeros(n)
for k in range(28):
    i = int(k * .07 * SR); m = int(.01 * SR); tick[i:i+m] += np.sin(2 * np.pi * 3000 * np.arange(m) / SR) * np.exp(-np.arange(m) / SR * 400)
add(rev * .3 + tick * .4, 36, .9)
# whiteout swell
n = int(1.6 * SR); add(filt(noise(1.6), 'lowpass', 3000) * np.linspace(0, 1, n) ** 2, 36.4, .4)

# ---------- CTA ding ----------
for f, d in [(880, 0), (1320, .12)]:
    n = int(.8 * SR); x = np.arange(n) / SR; add(np.sin(2 * np.pi * f * x) * env(n, .002, .25), 46 + d, .2)

# master
mix = np.stack([L, R], 1)
fade = np.ones(N); fl = int(1.2 * SR); fade[-fl:] = np.linspace(1, 0, fl); mix *= fade[:, None]
mix = np.tanh(mix * 1.1) * .9
mix /= max(1e-6, np.abs(mix).max()) / .9
pcm = (mix * 32767).astype(np.int16)
with wave.open('flash_audio.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('ok')
