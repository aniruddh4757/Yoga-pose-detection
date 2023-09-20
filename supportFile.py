import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import time
import os
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model2.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predict(img):
    # Replace this with the path to your image
    image = Image.open('static/images/test_image.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1   #(0 to 255  ==>> -1 to 1)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    idx = np.argmax(prediction)
    if prediction[0][idx]<0.75:
        return "Yoga Pose not recognized"

    if idx == 0:
        return "Downdog(Add Description here) : " + str(prediction[0][idx])
    elif idx == 1:
        return "Godess(Add Description here) : " + str(prediction[0][idx])
    elif idx == 2:
        return "Plank(Add Description here) : " + str(prediction[0][idx])
    elif idx == 3:
        return "Tree(Add Description here) : " + str(prediction[0][idx])
    elif idx == 4:
        return "Worrier(Add Description here) : " + str(prediction[0][idx])
    elif idx == 5:
        return "Chair(Add Description here) : " + str(prediction[0][idx])
  

def get_frame1(video):
	camera_port=video
	camera=cv2.VideoCapture(camera_port) #this makes a web cam object
	time.sleep(2)

	while True:
		ret, img = camera.read()
		cv2.imwrite(os.path.join("static/images/","test_image.jpg"),img)
		result = predict(img)

		cv2.putText(img, result, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))

		imgencode=cv2.imencode('.jpg',img)[1]
		stringData=imgencode.tostring()
		yield (b'--frame\r\n'
			b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

	del(camera)

