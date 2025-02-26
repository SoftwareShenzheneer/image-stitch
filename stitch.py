import numpy as np
import cv2
import glob
import imutils

imagePath = glob.glob("./images/*.jpg")
images = []

for image in imagePath:
    # Read the images one by one
    img = cv2.imread(image)
    images.append(img)

    # Because I like it, show the images individually
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 10, h // 10))
    cv2.imshow(image, img)
    cv2.waitKey(1)

# Create a stitcher through the API
imageStitcher = cv2.Stitcher.create()
error, stitchedImage = imageStitcher.stitch(images)

# If stitched successfully, process to remove borders
if not error:
    stitchedImage = cv2.copyMakeBorder(stitchedImage, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    gray = cv2.cvtColor(stitchedImage, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    x, y, w, h, = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    contours = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(areaOI)

    # Write processed image to an output
    stitchedImage = stitchedImage[y:y + h, x:x + w]
    cv2.imwrite("stitched.png", stitchedImage)

    # Display processed image
    h, w = stitchedImage.shape[:2]
    stitchedImage = cv2.resize(stitchedImage, (w // 10, h // 10))
    cv2.imshow("stitchedImage", stitchedImage)
    cv2.waitKey(0)

# Else the stitcher resulted in an error
else:
    print("Error stitching.")
    input()
