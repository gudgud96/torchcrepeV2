from torchcrepeV2 import ONNXTorchCrepePredictor
import librosa
import numpy as np
import matplotlib.pyplot as plt


def test_onnx():
    crepe_predictor = ONNXTorchCrepePredictor()
    y, sr = librosa.load(
        "tests/test_audio/vital_test_pluck_1.wav", 
        sr=16000, 
        duration=4
    )

    # ground truth from crepe
    y_ref = np.load("tests/test_audio/vital_test_pluck_1_f0.npy")

    # all arguments are aligned with original crepe implementation
    f = crepe_predictor.predict(
        audio=y, 
        sr=sr,
        viterbi=True, 
        center=True, 
        step_size=10
    )
    f = f[:-1]

    assert np.mean(np.abs(f - y_ref)) < 2e-4
    assert np.max(np.abs(f - y_ref)) < 1e-2