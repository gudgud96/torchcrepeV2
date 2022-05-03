import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided
from .crepe_model import TorchCrepe
from .utils import to_viterbi_cents, to_local_average_cents
import os

class TorchCrepePredictor:
    def __init__(self, device="cuda"):
        self.model = TorchCrepe()
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/model-full-crepe.pt")))
        self.model.eval()

        self.device = device
        if self.device == "cuda":
            self.model.cuda()

    def predict(self, audio, sr=16000, viterbi=True, center=True, step_size=10):
        """
        Perform pitch estimation on given audio

        Parameters
        ----------
        audio : np.ndarray [shape=(N,) or (N, C)]
            The audio samples. Multichannel audio will be downmixed.
        sr : int
            Sample rate of the audio samples. The audio will be resampled if
            the sample rate is not 16 kHz, which is expected by the model.
        model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
            String specifying the model capacity; see the docstring of
            :func:`~crepe.core.build_and_load_model`
        viterbi : bool
            Apply viterbi smoothing to the estimated pitch curve. False by default.
        center : boolean
            - If `True` (default), the signal `audio` is padded so that frame
            `D[:, t]` is centered at `audio[t * hop_length]`.
            - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
        step_size : int
            The step size in milliseconds for running pitch estimation.
        verbose : int
            Set the keras verbosity mode: 1 (default) will print out a progress bar
            during prediction, 0 will suppress all non-error printouts.

        Returns
        -------
        A 4-tuple consisting of:

            time: np.ndarray [shape=(T,)]
                The timestamps on which the pitch was estimated
            frequency: np.ndarray [shape=(T,)]
                The predicted pitch values in Hz
            confidence: np.ndarray [shape=(T,)]
                The confidence of voice activity, between 0 and 1
            activation: np.ndarray [shape=(T, 360)]
                The raw activation matrix
        """
        
        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        if center:
            x = np.pad(audio, 512, mode='constant', constant_values=0)
        hop_length = int(sr * step_size / 1000)     # step_size = int(1000 * 160 / 16000)
        n_frames = 1 + int((len(x) - 1024) / hop_length)
        frames = as_strided(x, shape=(1024, n_frames),
                            strides=(x.itemsize, hop_length * x.itemsize))
        frames = frames.transpose().copy()

        # normalize each frame -- this is expected by the model
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.std(frames, axis=1)[:, np.newaxis]

        frames = torch.tensor(frames)
        if self.device == "cuda":
            frames = frames.cuda()
        
        y = self.model(frames)

        if viterbi:
            cents = to_viterbi_cents(y.cpu().detach().numpy())
        else:
            cents = to_local_average_cents(y.cpu().detach().numpy())
        
        frequency = 10 * 2 ** (cents / 1200)
        frequency[np.isnan(frequency)] = 0

        return frequency