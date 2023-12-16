import numpy as np
from PIL import Image


def np_angle(a, b, c):
    # Landmarks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def predict_face_pose(mtcnn, path):
    frame = Image.open(path)
    # Convert the image if it has more than 3 channels, because MTCNN does not accept more than 3 channels.
    if frame.mode != "RGB":
        frame = frame.convert('RGB')

    # The detection part producing bounding box, probability of the detected face, and the facial landmarks
    bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True)
    angles_r = []
    angles_l = []
    orientations = []

    for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
        if bbox is not None:  # To check if we detect a face in the image
            if prob > 0.9:  # To check if the detected face has probability more than 90%, to avoid
                ang_r = np_angle(landmarks[0], landmarks[1], landmarks[2])  # Calculate the right eye angle
                ang_l = np_angle(landmarks[1], landmarks[0], landmarks[2])  # Calculate the left eye angle
                angles_r.append(ang_r)
                angles_l.append(ang_l)
                if (int(ang_r) in range(35, 57)) and (int(ang_l) in range(35, 58)):
                    orientation = 'Frontal'
                else:
                    if ang_r < ang_l:
                        orientation = 'Left Profile'
                    else:
                        orientation = 'Right Profile'
                orientations.append(orientation)
            else:
                print('The detected face is Less then the detection threshold')
        else:
            print('No face detected in the image')
    return landmarks_, angles_r, angles_l, orientations
