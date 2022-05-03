from .core import TorchCrepePredictor
import librosa
import matplotlib.pyplot as plt
import crepe
import tensorflow as tf
import numpy as np

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


y, sr = librosa.load("../../litmus/litmus/test_audio/kygo_pluck_3s.wav", sr=16000)
torch_crepe = TorchCrepePredictor()
f = torch_crepe.predict(y, sr)
length = y.shape[0] // 100
print(f.shape)

f0 = crepe.predict(
        y,
        sr,
        step_size=int(1000 * 100 / sr),
        verbose=1,
        center=True,
        viterbi=True,
        model_capacity="full"
)
f0 = f0[1]
print(f0.shape)

if f.shape[-1] != length:
    f = np.interp(
        np.linspace(0, 1, length, endpoint=False),
        np.linspace(0, 1, f.shape[-1], endpoint=False),
        f,
    )

if f0.shape[-1] != length:
    f0 = np.interp(
        np.linspace(0, 1, length, endpoint=False),
        np.linspace(0, 1, f0.shape[-1], endpoint=False),
        f0,
    )

plt.plot(f, label='torchcrepe')
plt.plot(f0, label='crepe')
plt.legend()
plt.show()
