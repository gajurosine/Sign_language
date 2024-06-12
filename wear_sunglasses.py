import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the sunglasses images
sunglasses_male = cv2.imread('sunglasses-kitty-02.png', cv2.IMREAD_UNCHANGED)
sunglasses_female = cv2.imread('sunglasses-kitty-02.png', cv2.IMREAD_UNCHANGED)

# Load the gender detection model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define a scaling factor for the sunglasses size
sunglasses_scale = 0.8  # Adjust as needed

# Gender classification threshold
gender_threshold = 0.6  # Adjust as needed

# Start the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Prepare the input image for gender detection
        face_roi = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Run gender detection
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()

        # Get the predicted gender and confidence
        gender = "Male" if genderPreds[0, 0] > gender_threshold else "Female"
        gender_confidence = genderPreds[0, 0]

        # Choose the sunglasses based on gender
        if gender == "Male":
            sunglasses = sunglasses_male
        else:
            sunglasses = sunglasses_female

        # Calculate the position and size of the sunglasses
        sunglasses_width = int(sunglasses_scale * w)
        sunglasses_height = int(sunglasses_width * sunglasses.shape[0] / sunglasses.shape[1])

        # Resize the sunglasses image
        sunglasses_resized = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))

        # Calculate the position to place the sunglasses
        x1 = x + int(w/2) - int(sunglasses_width/2)
        x2 = x1 + sunglasses_width
        y1 = y + int(0.55 * h) - sunglasses_height  
        y2 = y1 + sunglasses_height

        # Adjust for out-of-bounds positions
        x1 = max(x1, 0)
        x2 = min(x2, frame.shape[1])
        y1 = max(y1, 0)
        y2 = min(y2, frame.shape[0])

        # Create a mask for the sunglasses
        sunglasses_mask = sunglasses_resized[:, :, 3] / 255.0
        frame_roi = frame[y1:y2, x1:x2]

        # Blend the sunglasses with the frame
        for c in range(0, 3):
            frame_roi[:, :, c] = (1.0 - sunglasses_mask) * frame_roi[:, :, c] + sunglasses_mask * sunglasses_resized[:, :, c]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label = f"{gender}: {gender_confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Frame with Sunglasses', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
