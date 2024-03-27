from keras.models import load_model
import numpy as np

# model_path = "static/model/mfcc20_conv1d.keras"
model_path = "static/model/mfcc_new_130_conv1d.keras"

model = load_model(model_path)


def custom_argmax(arr, axis=1):
    if axis == 0:
        raise ValueError("Axis 0 is not supported for custom_argmax function.")
    max_indices = []
    for row in arr:
        max_val = float("-inf")
        max_index = None
        for i, val in enumerate(row):
            if val > max_val:
                max_val = val
                max_index = i
        max_indices.append(max_index)
    return np.array(max_indices)


def custom_most_common(arr):
    # Create a dictionary to store the counts of each element
    counts = {}
    # Iterate through the array
    for element in arr:
        # If the element is already in the dictionary, increment its count
        if element in counts:
            counts[element] += 1
        # Otherwise, initialize its count to 1
        else:
            counts[element] = 1
    # Find the element with the maximum count
    most_common_element = None
    max_count = 0
    for element, count in counts.items():
        if count > max_count:
            max_count = count
            most_common_element = element
    return most_common_element


def custom_mean(predictions, predicted_genre_index):
    total_sum = 0
    count = 0
    # Iterate through the rows of the predictions list
    for row in predictions:
        # Check if the row index is within the bounds of the list
        if predicted_genre_index < len(row):
            # Add the value at the predicted_genre_index to the total sum
            total_sum += row[predicted_genre_index]
            count += 1
    # Calculate the mean
    if count != 0:
        mean = total_sum / count
    else:
        mean = None
    return mean


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
    print(predictions)

    # Get the index of the most confident prediction for each segment
    # predicted_classes = np.argmax(predictions, axis=1)
    predicted_classes = custom_argmax(predictions, axis=1)
    print(predicted_classes)

    # Count occurrences of each predicted class and get the most frequent one
    # predicted_genre_index = np.bincount(predicted_classes).argmax()
    predicted_genre_index = custom_most_common(predicted_classes)
    print(predicted_genre_index)

    # Get the corresponding genre label from the mapping
    predicted_genre = genre_mapping[predicted_genre_index]

    # Calculate the confidence score as the mean probability of the predicted class
    # confidence_score = np.mean(predictions[:, predicted_genre_index])
    confidence_score = custom_mean(predictions, predicted_genre_index)
    print(confidence_score)

    return predicted_genre, confidence_score
