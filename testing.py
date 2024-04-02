# import os
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np

# # Load the trained model
# model = load_model('posture_detection_model.h5')

# # Directory path for new images
# new_images_dir = 'path/to/new_images_folder'

# # Preprocess and predict on new images
# for img_name in os.listdir(new_images_dir):
#     img_path = os.path.join(new_images_dir, img_name)
#     img = image.load_img(img_path, target_size=(img_width, img_height))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img /= 255.0
#     prediction = model.predict(img)
#     if prediction[0][0] > 0.5:
#         print(f"{img_name}: Good Posture")
#     else:
#         print(f"{img_name}: Bad Posture (Too Close to the Screen)")
print(36//8)
