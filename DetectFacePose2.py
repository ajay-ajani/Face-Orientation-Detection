import os
import torch
import shutil
import FacePoseDetector

from facenet_pytorch import MTCNN

left_offset = 20
fontScale = 2
fontThickness = 3
text_color = (0, 0, 255)
lineColor = (255, 255, 0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

mtcnn = MTCNN(image_size=160,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
              factor=0.709,
              post_process=True,
              device=device  # If you don't have GPU
              )

pics = []
dir_name = "/Users/ajani/Downloads/"

files = os.listdir()
for file in files:
    try:
        _, _, _, orientations = FacePoseDetector.predict_face_pose(mtcnn, dir_name + file)
        count = len(orientations)
        if count == 0:
            print(f"ERROR: {file}. No face detected.")
        elif count == 1:
            orientation = orientations[0]
            if orientation == 'Frontal':
                pics.append(file)
            else:
                print(f"ERROR: {file}. Orientation is not frontal.")
        else:
            print(f"ERROR: {file}. More than one face detected")
    except Exception as e:
        print(f"ERROR: {file}. Issue with image path: {e}")
