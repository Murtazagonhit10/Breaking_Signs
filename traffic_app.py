import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Configuration ---
# The model was likely trained on 32x32 images to produce 4096 features
IMAGE_SIZE = (48, 48)
MODEL_PATH = "best_traffic_sign_model.keras"
NUM_CLASSES = 43  # Based on GTSRB dataset

# Dummy placeholder for sign names. You MUST replace this with your actual sign names.
SIGN_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Trucks Prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Icy/snowy road",
    31: "Wild animals crossing",
    32: "End of all speed and no-passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over trucks",
}


@st.cache_resource
def load_keras_model():
    """Loads the pre-trained Keras model from disk."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the PIL image to match the model's expected input shape and format.

    The crucial change is resizing to IMAGE_SIZE (32x32) to match the dimensions
    that resulted in the 4096 features the dense layer expects.
    """
    # 1. Resize the image to the size used for training (32x32)
    image = image.resize(IMAGE_SIZE)

    # 2. Convert PIL Image to NumPy array
    img_array = np.array(image)

    # 3. Normalize the pixel values (Assuming the model was trained on [0, 1] normalized data)
    img_array = img_array / 255.0

    # 4. Expand dimensions to create a batch of size 1 (required for model.predict)
    # Shape changes from (32, 32, 3) to (1, 32, 32, 3)
    processed_input = np.expand_dims(img_array, axis=0)

    # Check the processed input shape (for debugging in a real app)
    # st.sidebar.text(f"Processed input shape: {processed_input.shape}")

    return processed_input


def main():
    st.title("🚦 German Traffic Sign Classifier")
    st.markdown(
        "Upload an image of a German traffic sign for classification using a Convolutional Neural Network."
    )

    # Load the model
    model = load_keras_model()

    if model is None:
        st.warning(
            f"Prediction requires the model file. Please ensure the file is named `{MODEL_PATH}` and is in the same directory."
        )
        return

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "ppm"],
        help="Upload an image of a German Traffic Sign.",
    )

    if uploaded_file is not None:
        # 1. Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Sign")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown(
                f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; border-left: 5px solid #ff4b4b;'>
                    **File Details:**<br>
                    Format: {image.format}<br>
                    Original Size: {image.size[0]}x{image.size[1]} pixels
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Preprocess and Predict
            if st.button("Classify Sign", type="primary"):
                st.write("---")

                # Show loading state
                with st.spinner("Analyzing image and predicting sign..."):
                    try:
                        # Preprocess the image (Resizes to 32x32)
                        processed_input = preprocess_image(image)

                        # Make prediction
                        predictions = model.predict(processed_input)

                        # Get the predicted class index
                        predicted_class_index = np.argmax(predictions, axis=1)[0]
                        predicted_sign_name = SIGN_NAMES.get(
                            predicted_class_index,
                            f"Class ID {predicted_class_index} (Name unknown)",
                        )

                        # Get the confidence score
                        confidence = predictions[0][predicted_class_index] * 100

                    except ValueError as e:
                        # This catches any remaining shape errors, but the resize should fix the original one.
                        st.error(
                            f"""
                        **Prediction Error:** The model is still expecting a different input shape. 
                        Received shape: {processed_input.shape}. Error: {e}
                        This usually means the image size in `preprocess_image` still doesn't match the size 
                        the saved model was trained on. Try a different resize dimension in `IMAGE_SIZE`.
                        """
                        )
                        return
                    except Exception as e:
                        st.error(f"An unexpected error occurred during prediction: {e}")
                        return

                # 3. Display Results
                st.subheader("Prediction Result")
                st.markdown(
                    f"""
                    <div style='background-color: #e6ffe6; padding: 15px; border-radius: 10px; border-left: 5px solid #00c853;'>
                        The predicted traffic sign is: **{predicted_sign_name}**
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                # Optional: Show all probabilities for a deeper look
                with st.expander("Show Detailed Probabilities"):
                    # Get top 5 predictions
                    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                    top_5_probabilities = predictions[0][top_5_indices] * 100

                    top_results = {
                        SIGN_NAMES.get(i, f"Class ID {i}"): f"{p:.2f}%"
                        for i, p in zip(top_5_indices, top_5_probabilities)
                    }
                    st.dataframe(
                        pd.DataFrame(
                            top_results.items(), columns=["Traffic Sign", "Confidence"]
                        )
                    )


if __name__ == "__main__":
    # Import pandas only if needed for the expanded view, otherwise it's in a try block
    try:
        import pandas as pd
    except ImportError:
        st.warning(
            "Please install pandas (`pip install pandas`) to show detailed probabilities."
        )

    # Set up basic page config
    st.set_page_config(page_title="Traffic Sign Classifier", layout="wide")

    # Check for TensorFlow/Keras version compatibility if needed, but proceeding with main
    main()
