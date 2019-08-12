import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('test.jpg')

scale_percent = 300 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


# Convert into grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#Count faces
print("Number of faces: {}".format(len(faces)) )

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
# cv2.resizeWindow('image', 600,600)

cv2.imshow('img', resized)
cv2.waitKey()
