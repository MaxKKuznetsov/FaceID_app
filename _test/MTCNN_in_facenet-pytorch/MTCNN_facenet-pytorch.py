from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

help(MTCNN)

# Create face detector
mtcnn = MTCNN(select_largest=False, device='cuda')

# Load a single image and display
v_cap = cv2.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/agqphdxmwt.mp4')
success, frame = v_cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = Image.fromarray(frame)

plt.figure(figsize=(12, 8))
plt.imshow(frame)
plt.axis('off')

# Detect face
face = mtcnn(frame)
face.shape