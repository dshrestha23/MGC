from keras.models import load_model
import numpy as np

model_path = "static/model/cnn_mfcc_1D.keras"

model = load_model(model_path)


def predict(preprocessed_audio):
    # Manual genre mapping
    genre_mapping = {
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
    predictions = model.predict(preprocessed_audio)
    mean_probabilities = np.mean(predictions, axis=0)
    most_likely_genre_index = np.argmax(mean_probabilities)

    # Get the overall predicted genre
    overall_predicted_genre = genre_mapping[most_likely_genre_index]

    return overall_predicted_genre
