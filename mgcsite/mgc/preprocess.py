# Imports

import librosa
import numpy as np
import scipy.fftpack as fft
import librosa.filters
import math
from .pre_func import generate_mfcc

# from .pre_func import stft as stf

# Manual genre mapping
genre1_mapping = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}
segment_duration = 20
overlap = 0.5
SR = 22050


def reshape_input(mfcc_features):
    mfcc1_features = [
        mfcc.reshape((mfcc.shape[0], mfcc.shape[1], 1)) for mfcc in mfcc_features
    ]
    return mfcc1_features


def preprocess(audio, SR=SR, segment_duration=segment_duration, overlap=overlap):
    # Initialize empty list to store features and corresponding labels
    mfcc_features = []
    try:
        # Load the audio file
        y, sr = librosa.load(audio, sr=SR)
        # Extract MFCC features
        # mfcc = generate_mfcc(y, sr)

        # Append the MFCC features and corresponding label
        # mfcc_features.append(mfcc)

        # Calculate the number of samples per segment
        segment_samples = int(segment_duration * sr)

        # Calculate the number of samples to overlap
        # overlap_samples = int(overlap * segment_samples)

        # Extract MFCC features for each segment
        # for i in range(
        #     0, len(y) - segment_samples + 1, segment_samples - overlap_samples
        # ):
        if len(y) <= segment_samples:
            mfcc = generate_mfcc(y, sr)
            mfcc_features.append(mfcc)
        else:

            # for i in range(0, len(y) - segment_samples + 1, segment_samples):
            for i in range(0, segment_samples, segment_samples):
                segment = y[i : i + segment_samples]

                # Extract MFCC features
                mfcc = generate_mfcc(y, sr)

                # Append the MFCC features and corresponding label
                mfcc_features.append(mfcc)

    except Exception as e:
        print(f"Error processing {audio}: {e}")

    print("MFCC FEATURES: ", mfcc_features)
    print(mfcc_features[0].shape)
    mfcc1_features = reshape_input(mfcc_features)
    # print("MFCC1 FEATURES: ", mfcc1_features)
    mfcc1_features = np.array(mfcc_features)
    return mfcc1_features
