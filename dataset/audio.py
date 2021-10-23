import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size(args):
    hop_size = args.hop_size
    if hop_size is None:
        assert args.frame_shift_ms is not None
        hop_size = int(args.frame_shift_ms / 1000 * args.sample_rate)
    return hop_size

def linearspectrogram(args, wav):
    D = _stft(preemphasis(wav, args.preemphasis, args.preemphasize))
    S = _amp_to_db(np.abs(D)) - args.ref_level_db
    
    if args.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(args, wav):
    x = preemphasis(wav, args.preemphasis, args.preemphasize)
    D = _stft(args, x)
    x = _linear_to_mel(args, np.abs(D))
    S = _amp_to_db(args, x) - args.ref_level_db
    
    if args.signal_normalization:
        return _normalize(args, S)
    return S

def _lws_processor(args):
    import lws
    return lws.lws(args.n_fft, get_hop_size(args), fftsize=args.win_size, mode="speech")

def _stft(args, y):
    if args.use_lws:
        return _lws_processor(args).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=args.n_fft, hop_length=get_hop_size(args), win_length=args.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(args, spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(args)
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis(args):
    assert args.fmax <= args.sample_rate // 2
    return librosa.filters.mel(args.sample_rate, args.n_fft, n_mels=args.num_mels,
                               fmin=args.fmin, fmax=args.fmax)

def _amp_to_db(args, x):
    min_level = np.exp(args.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(args, S):
    if args.allow_clipping_in_normalization:
        if args.symmetric_mels:
            return np.clip((2 * args.max_abs_value) * ((S - args.min_level_db) / (-args.min_level_db)) - args.max_abs_value,
                           -args.max_abs_value, args.max_abs_value)
        else:
            return np.clip(args.max_abs_value * ((S - args.min_level_db) / (-args.min_level_db)), 0, args.max_abs_value)
    
    assert S.max() <= 0 and S.min() - args.min_level_db >= 0
    if args.symmetric_mels:
        return (2 * args.max_abs_value) * ((S - args.min_level_db) / (-args.min_level_db)) - args.max_abs_value
    else:
        return args.max_abs_value * ((S - args.min_level_db) / (-args.min_level_db))

def _denormalize(args, D):
    if args.allow_clipping_in_normalization:
        if args.symmetric_mels:
            return (((np.clip(D, -args.max_abs_value,
                              args.max_abs_value) + args.max_abs_value) * -args.min_level_db / (2 * args.max_abs_value))
                    + args.min_level_db)
        else:
            return ((np.clip(D, 0, args.max_abs_value) * -args.min_level_db / args.max_abs_value) + args.min_level_db)
    
    if args.symmetric_mels:
        return (((D + args.max_abs_value) * -args.min_level_db / (2 * args.max_abs_value)) + args.min_level_db)
    else:
        return ((D * -args.min_level_db / args.max_abs_value) + args.min_level_db)
