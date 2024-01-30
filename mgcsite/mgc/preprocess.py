# Imports

import librosa
import numpy as np


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
segment_duration = 3
overlap = 0.5
SR = 22050


def generate_mfcc(segment, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    mfcc = mfcc.T
    return mfcc


def reshape_input(mfcc_features):
    mfcc1_features = [
        mfcc.reshape((mfcc.shape[0], mfcc.shape[1], 1)) for mfcc in mfcc_features
    ]
    return mfcc1_features


def normalize(mfcc1_features):
    mfcc1_features = (mfcc1_features - np.mean(mfcc1_features)) / np.std(mfcc1_features)
    return mfcc1_features


def preprocess(audio, SR=SR, segment_duration=segment_duration, overlap=overlap):
    # Initialize empty list to store features and corresponding labels
    mfcc_features = []
    try:
        # Load the audio file
        y, sr = librosa.load(audio, sr=SR)

        # Calculate the number of samples per segment
        segment_samples = int(segment_duration * sr)

        # Calculate the number of samples to overlap
        overlap_samples = int(overlap * segment_samples)

        # Extract MFCC features for each segment
        for i in range(
            0, len(y) - segment_samples + 1, segment_samples - overlap_samples
        ):
            segment = y[i : i + segment_samples]

            # Extract MFCC features
            mfcc = generate_mfcc(segment, sr)

            # Append the MFCC features and corresponding label
            mfcc_features.append(mfcc)

    except Exception as e:
        print(f"Error processing {audio}: {e}")

    mfcc1_features = reshape_input(mfcc_features)
    mfcc1_features = np.array(mfcc1_features)
    mfcc1_features = normalize(mfcc1_features)

    return mfcc1_features
