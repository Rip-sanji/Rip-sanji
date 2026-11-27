import subprocess, wave, numpy as np, os, sys
from scipy.signal import resample_poly, butter, sosfilt
D = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(D, 'pt-br-edresson-low.onnx')
SEGS = [
    (0.15, "E se a sua velocidade dobrasse... a cada segundo?"),
    (4.6, "No começo, tudo normal: cinco por hora... dez... vinte..."),
    (8.5, "Passou o Iussein Bolt!"),
    (10.05, "Quatro segundos: oitenta por hora!"),
    (13.05, "Seis segundos: Fórmula Um."),
    (15.05, "E não para de dobrar..."),
    (17.0, "Ele quebra a barreira do som!"),
    (19.0, "O ar pega fogo!"),
    (20.65, "Agora vai! Ele sai da cidade..."),
    (22.7, "e corre por cima da água, explodindo tudo atrás dele!"),
    (26.1, "Treze segundos: mais rápido que um foguete."),
    (28.5, "Agora ele dá voltas na Terra... igual o Flash."),
    (31.1, "Até que chega perto da velocidade da luz..."),
    (34.7, "E quando passa dela... o tempo começa a andar pra trás."),
    (38.3, "Ele volta pro passado... e passa do lado... dele mesmo."),
    (41.1, "Sou eu?!"),
    (42.7, "Era ele no começo do vídeo."),
    (44.8, "E você? O que faria com essa velocidade? Comenta aí! E segue pra mais, E se."),
]
SR = 44100
PITCH = 0.79  # <1 = voz mais grossa

def synth(text, ls):
    out = os.path.join(D, 'seg.wav')
    subprocess.run([sys.executable, '-m', 'piper', '-m', MODEL, '-f', out, '--length-scale', str(ls), '--sentence-silence', '0.1'],
                   input=text.encode(), check=True, capture_output=True)
    deep = os.path.join(D, 'seg_deep.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', out, '-af', f'asetrate={16000*PITCH:.0f},aresample=16000,atempo={1/PITCH:.4f}', deep], check=True)
    out = deep
    with wave.open(out) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        sr = w.getframerate()
    # trim silence
    idx = np.where(np.abs(x) > 0.02)[0]
    if len(idx): x = x[max(0, idx[0] - 200): idx[-1] + 800]
    return resample_poly(x, SR, sr)

dur = 50.0
voice = np.zeros(int(dur * SR))
for k, (t0, text) in enumerate(SEGS):
    slot = (SEGS[k + 1][0] if k + 1 < len(SEGS) else 49.6) - t0 - 0.08
    ls = 0.92
    x = synth(text, ls)
    while len(x) / SR > slot and ls > 0.62:
        ls -= 0.05; x = synth(text, ls)
    print(f'{t0:5.2f}s ls={ls:.2f} {len(x)/SR:.2f}/{slot:.2f}s  {text}')
    i = int(t0 * SR); voice[i:i + len(x)] += x[:len(voice) - i]

# voice EQ: cut rumble, presence boost, gentle compression
voice = sosfilt(butter(2, 90, 'highpass', fs=SR, output='sos'), voice)
pres = sosfilt(butter(2, [2500, 6000], 'bandpass', fs=SR, output='sos'), voice)
bass = sosfilt(butter(2, 220, 'lowpass', fs=SR, output='sos'), voice)
voice = voice + 0.35 * pres + 0.6 * bass
voice = np.tanh(voice / (np.abs(voice).max() + 1e-9) * 2.2) * 0.95

with wave.open(os.path.join(D, '../vid/flash_audio.wav')) as w:
    m = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32).reshape(-1, 2) / 32768
m = m[:len(voice)]
# ducking envelope
env = np.abs(voice)
win = int(0.25 * SR); env = np.convolve(env, np.ones(win) / win, 'same')
duck = 1 - 0.55 * np.clip(env / (env.max() * 0.25), 0, 1)
mix = m * duck[:len(m), None] * 0.8 + voice[:len(m), None] * 0.9
mix /= np.abs(mix).max() / 0.95
with wave.open(os.path.join(D, 'mix.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
with wave.open(os.path.join(D, 'narracao_so_voz.wav'), 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((voice * 32767).astype(np.int16).tobytes())
print('ok')
