import librosa
import numpy as np
import scipy.fftpack as fft
import librosa.filters
import math


def custom_hanning(n):
    """
    Custom Hanning window function without using any library.

    Args:
    n (int): Length of the window.

    Returns:
    list: Hanning window of length n.
    """
    hanning_window = []
    for i in range(n):
        angle = 2 * 3.14159265358979323846 * i / (n - 1)
        value = 0.5 - 0.5 * (
            0.5 - 0.5 * (1 - (2 * angle / 3.14159265358979323846) ** 2)
        )
        hanning_window.append(value)
    return hanning_window


def custom_fft(frame, n_fft):
    """
    Custom Fast Fourier Transform (FFT) function without using any library.

    Args:
    frame (np.ndarray): Input audio frame.
    n_fft (int): Number of FFT points.

    Returns:
    np.ndarray: FFT result.
    """
    M = len(frame)
    w = np.exp(-2j * np.pi / M)

    # Cooley-Tukey FFT algorithm
    if M <= 1:
        return frame
    even_fft = custom_fft(frame[::2], n_fft)
    odd_fft = custom_fft(frame[1::2], n_fft)
    terms = np.exp(-2j * np.pi * np.arange(M) / M) * w
    # return np.concatenate(
    #     [even_fft + terms[: M // 2] * odd_fft, even_fft + terms[M // 2 :] * odd_fft]
    # )
    # Manually concatenate arrays
    concatenated = []
    for i in range(len(even_fft)):
        concatenated.append(even_fft[i] + terms[i] * odd_fft[i])

    for i in range(len(odd_fft), len(even_fft)):
        concatenated.append(even_fft[i] + terms[i] * odd_fft[i])

    # Convert the concatenated list to an array
    result = np.array(concatenated)
    return result


def custom_stft(x, n_fft, hop_length):
    """
    Custom Short-Time Fourier Transform (STFT) function without using any library.

    Args:
    x (np.ndarray): Input audio signal.
    n_fft (int): Number of FFT points.
    hop_length (int): Hop length for the STFT.

    Returns:
    np.ndarray: Magnitude spectrogram.
    """
    # Get the length of the input signal
    x_len = len(x)

    # Calculate the number of frames
    num_frames = 1 + (x_len - n_fft) // hop_length

    # Precompute Hanning window
    window = np.hanning(n_fft)

    # Preallocate the spectrogram matrix
    stft_matrix = np.empty((n_fft // 2 + 1, num_frames), dtype=np.float64)

    # Iterate over the frames
    for i in range(num_frames):
        # Extract the current frame
        start = i * hop_length
        end = start + n_fft
        frame = x[start:end]

        # Apply windowing (Hanning window)
        frame *= window

        # Compute FFT manually
        # fft_result = custom_fft(frame, n_fft)
        fft_result = np.fft.fft(frame, n=n_fft)

        # Store the magnitude spectrum of the current frame
        stft_matrix[:, i] = np.abs(fft_result)[: n_fft // 2 + 1]

    return stft_matrix


def custom_mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """
    Custom function to calculate the mel filterbank.

    Args:
    sr (number): Sampling rate of the audio signal.
    n_fft (int): Number of FFT points.
    n_mels (int): Number of mel bands to generate.
    fmin (number): Minimum frequency in Hz.
    fmax (number): Maximum frequency in Hz.

    Returns:
    np.ndarray: Mel filterbank matrix.
    """
    # Define the frequency range
    # Handle None values for fmin and fmax
    if fmax is None:
        fmax = sr / 2  # Nyquist frequency
    freqs = np.linspace(fmin, fmax, n_fft // 2 + 1)

    # Convert frequencies to mel scale
    mel_fmin = 2595 * np.log10(1 + fmin / 700)
    mel_fmax = (
        2595 * np.log10(1 + fmax / 700) if fmax else 2595 * np.log10(1 + sr / 2 / 700)
    )
    mel_freqs = np.linspace(mel_fmin, mel_fmax, n_mels + 2)

    # Convert mel frequencies back to Hz scale
    hz_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)

    # Create the mel filterbank matrix
    mel_filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        for j in range(n_fft // 2 + 1):
            if freqs[j] < hz_freqs[i]:
                mel_filterbank[i, j] = 0
            elif freqs[j] <= hz_freqs[i + 1]:
                mel_filterbank[i, j] = (freqs[j] - hz_freqs[i]) / (
                    hz_freqs[i + 1] - hz_freqs[i]
                )
            elif freqs[j] <= hz_freqs[i + 2]:
                mel_filterbank[i, j] = (hz_freqs[i + 2] - freqs[j]) / (
                    hz_freqs[i + 2] - hz_freqs[i + 1]
                )

    return mel_filterbank


def apply_mel_filterbank_custom(mel_basis, stft):
    """
    Apply the mel filterbank to the STFT.

    Args:
    mel_basis (np.ndarray): Mel filterbank matrix.
    stft (np.ndarray): Short-Time Fourier Transform matrix.

    Returns:
    np.ndarray: Mel spectrogram.
    """
    # Get the number of mel filters
    n_mels = mel_basis.shape[0]

    # Get the number of frames
    num_frames = stft.shape[1]

    # Initialize an empty array to store the mel spectrogram
    mel_spec = np.zeros((n_mels, num_frames))

    # Apply the mel filterbank to the STFT
    for i in range(n_mels):
        for j in range(num_frames):
            mel_spec[i, j] = np.dot(mel_basis[i], stft[:, j])

    return mel_spec


def custom_log(x):
    """
    Custom logarithm function.

    Args:
    x (float or array-like): Input value(s) to take the logarithm of.

    Returns:
    float or array-like: Result of the logarithm operation.
    """
    # Define a small constant to avoid taking the logarithm of zero or negative values
    eps = 1e-6

    # Handle scalar input
    if isinstance(x, (int, float)):
        if x <= 0:
            return -np.inf
        else:
            result = 0.0
            while x < 1:
                x *= 10
                result -= 2.3025850929940455  # log(10)
            while x >= 10:
                x /= 10
                result += 2.3025850929940455  # log(10)
            return result

    # Handle array-like input
    elif isinstance(x, (list, tuple)):
        result = []
        for val in x:
            if val <= 0:
                result.append(-np.inf)
            else:
                log_val = 0.0
                while val < 1:
                    val *= 10
                    log_val -= 2.3025850929940455  # log(10)
                while val >= 10:
                    val /= 10
                    log_val += 2.3025850929940455  # log(10)
                result.append(log_val)
        return result

    else:
        raise ValueError("Input must be a scalar, list, or tuple")


def custom_dct(matrix):
    """
    Custom Discrete Cosine Transform (DCT) function without using any library.

    Args:
    matrix (list of lists): Input matrix.

    Returns:
    list of lists: DCT result.
    """
    M = len(matrix)
    N = len(matrix[0])

    dct_result = []

    for k in range(M):
        dct_row = []
        for n in range(N):
            sum_val = 0.0
            for m in range(M):
                for l in range(N):
                    sum_val += (
                        matrix[m][l]
                        * np.cos(np.pi * k * (2 * l + 1) / (2 * M))
                        * np.cos(np.pi * n * (2 * m + 1) / (2 * N))
                    )
            dct_row.append(sum_val)
        dct_result.append(dct_row)

    return dct_result


def generate_mfcc(segment, sr, n_mfcc=20, n_fft=2048, hop_length=512, n_frames=130):

    print("Start STFT")
    stft = custom_stft(x=segment, n_fft=n_fft, hop_length=hop_length)
    print("Stop  STFT")

    # Step 2: Calculate the mel filterbank
    # mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=20, fmin=0, fmax=None)
    print("Start Mel filterbank ")
    mel_basis = custom_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=128, fmin=0, fmax=None)
    print("Stop Mel filterbank")

    # Step 3: Apply the mel filterbank to the STFT
    # mel_spec = np.dot(mel_basis, stft)
    print("START  Mel STFT")
    mel_spec = apply_mel_filterbank_custom(mel_basis, stft)
    print("STOP Mel STFT")

    # Step 4: Take the logarithm of the mel spectrogram
    # log_mel_spec = np.log(mel_spec + 1e-6)
    # log_mel_spec = custom_log(mel_spec + 1e-6)
    print("START  Log Transform")
    log_mel_spec = [[custom_log(val + 1e-6) for val in row] for row in mel_spec]
    print("Stop    Log Transform")

    # Step 5: Compute the discrete cosine transform (DCT)
    print("FFt start")
    mfcc = fft.dct(log_mel_spec, axis=0, type=2, norm="ortho")
    # mfcc = custom_dct(matrix=log_mel_spec)
    print("FFt end")

    # Keep only the first n_mfcc coefficients
    mfcc = mfcc[:n_mfcc]

    # Normalize MFCC features
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    # Transpose the MFCC matrix
    mfcc = mfcc.T
    # Pad or truncate the MFCC matrix to have n_frames frames
    if mfcc.shape[0] < n_frames:
        mfcc = np.pad(
            mfcc,
            ((0, n_frames - mfcc.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        mfcc = mfcc[:n_frames]

    return mfcc
