import pandas as pd
import os
import re
from textblob import TextBlob
import nltk
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input #type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type: ignore
import numpy as np
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore

# Download corpora for TextBlob
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Load captions file
captions_file = "captions.txt"
data = []

with open(captions_file, "r") as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 2:
            image, caption = parts
            data.append((image.split("#")[0], caption))

df = pd.DataFrame(data, columns=["image", "caption"])
df["image"] = "data/Images/" + df["image"]  # Corrected path

# Check for missing images and exclude them
df["image_exists"] = df["image"].apply(lambda x: os.path.exists(x))
df = df[df["image_exists"]]  # Only keep rows where the image exists
df.drop(columns=["image_exists"], inplace=True)

# Debugging: Print the number of valid images
print(f"Number of valid images after path check: {len(df)}")

# Preprocess captions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

df["cleaned_caption"] = df["caption"].apply(preprocess_text)

# Add sentiment labels
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["cleaned_caption"].apply(get_sentiment)

# Load image feature extraction model
vgg_model = VGG16(weights="imagenet", include_top=False)

def extract_image_features(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Resize to VGG16 input size
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = img_array.reshape(1, 224, 224, 3)  # Add batch dimension
        features = vgg_model.predict(img_array)
        return features.flatten()  # Flatten the feature vector
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None  # Skip if any error occurs

def extract_text_features(caption):
    """
    Extracts text features from a caption.
    Currently, it returns the sentiment polarity as a single feature.
    """
    blob = TextBlob(caption)
    sentiment_score = blob.sentiment.polarity  # Sentiment score ranges from -1 to 1
    return [sentiment_score]  # Return as a list for compatibility with np.concatenate

image_features_list = []
text_features_list = []
y = []  # Target labels for the model (sentiment)

# Loop through the dataframe to extract features
for _, row in df.iterrows():
    image_path = row["image"]
    caption = row["cleaned_caption"]
    
    # Extract image features
    image_feat = extract_image_features(image_path)
    if image_feat is None:  # Skip invalid images
        continue
    
    # Extract text features
    text_feat = extract_text_features(caption)
    
    # Append valid features
    image_features_list.append(image_feat)
    text_features_list.append(text_feat)
    
    # Add the sentiment as a label (0 for Negative, 1 for Positive)
    sentiment = row["sentiment"]
    if sentiment == "Positive":
        y.append(1)
    else:
        y.append(0)  # Assuming binary classification (0 for Negative, 1 for Positive)

# Debugging: Check the extracted feature counts
print(f"Image features: {len(image_features_list)}")
print(f"Text features: {len(text_features_list)}")
print(f"Labels: {len(y)}")

# Ensure feature and label arrays are aligned before creating X_combined
if image_features_list and text_features_list and len(image_features_list) == len(y):
    X_combined = np.hstack([np.array(image_features_list), np.array(text_features_list)])
else:
    print("Mismatch in features and labels. Exiting.")
    X_combined = None

if X_combined is not None:
    # Define the model
    model = Sequential([
        Dense(256, activation="relu", input_dim=X_combined.shape[1]),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")  # Binary output (Positive/Negative)
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X_combined, np.array(y), epochs=10, batch_size=32)
else:
    print("Exiting: No valid data for training.")
