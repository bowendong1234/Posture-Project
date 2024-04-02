from keras.models import load_model
import numpy as np
from keras.preprocessing import image

img_width, img_height = 960, 540

# Load the trained model
model = load_model('posture_detection_model.h5')  # Replace 'posture_detection_model.h5' with your model file path

# Preprocess the image
def preprocess_image(image_path, target_size=(img_width, img_height)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values
    return img

# Make predictions on a new image
def predict_posture(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        return "Good Posture"
    else:
        return "Bad Posture"

# Example usage
image_path = 'path/to/your/image.jpg'  # Replace with the path to your image
prediction = predict_posture(image_path)
print("Prediction:", prediction)
