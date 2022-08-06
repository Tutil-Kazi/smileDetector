import cv2

# classifiers for the face and smile detectors
detectFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detectSmile = cv2.CascadeClassifier('haarcascade_smile.xml')

# capture image from camera
camera = cv2.VideoCapture(0)

# show real time frames
while True:
    # read real time frames from the camera
    successful_frame_read, frame = camera.read()

    # if error occurs, exit
    if not successful_frame_read:
        break

    # converts RGB image to grayscale for optimization
    grayscaleImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the whole frame
    faces = detectFace.detectMultiScale(grayscaleImg)

    # run face detection loop on each face rectangles on whole frame
    for (x, y, w, h) in faces:

        # Draw rectangles
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 5)

        # subface forming by using numpy array slicing the whole frame
        face = frame[y:y+h, x:x+w]

        # converts RGB image to grayscale for optimization
        grayFace = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # scaleFactor is blurring the image and minNeighbors are number of neighboring smiles in the frame
        smiles = detectSmile.detectMultiScale(grayscaleImg, scaleFactor=1.8, minNeighbors=25)

        # detect smiles only on the face
        for (x_, y_, w_, h_) in smiles:

            # Draw rectangles of smile
            cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 5)

        # smiling label around the face
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling face', (x, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(255, 0, 255))

    # show real time frame
    cv2.imshow("Smile detector", frame);

    # display the image with a delay
    cv2.waitKey(10);

# release the memory
camera.release();
cv2.destroyAllWindows()
