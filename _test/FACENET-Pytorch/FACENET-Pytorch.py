# https://www.programmersought.com/article/84588005696/

# Making a database of face feature vectors Finally, two files are saved, namely the face feature vector and the corresponding name in the database. Of course, you can also save together
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
# INCEPTIONRESNETV1 provides two pre-training models, which are trained on the Vggface dataset and CASIA data sets, respectively.
#         If you do not download it, you may be very slow, you can download it from the author's Google Cloud link, then put it into C: \ Users \ Your username \ .cache \ Torch \ Checkpoints This folder
# If it is a Linux system, store it in / home / your username /.Cache/torch/checkpoints
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]


# Place all single photo images in their respective folders, the folder name is the name of the person, the storage format is as follows
'''
--orgin
  |--zhangsan
     |--1.jpg
     |--2.jpg
  |--lisi
     |--1.jpg
     |--2.jpg
'''
dataset = datasets.ImageFolder('./database/orgin')  #
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
aligned = []  # aligned is the face, the size of the face, the size is the image_size = 160 defined before.
names = []
i = 1
for x, y in loader:
    path = './database/aligned/{}/'.format(dataset.idx_to_class[y])  # This is the face path to save.
    if not os.path.exists(path):
        i = 1
        os.mkdir(path)
    # If you want to save the identified face, you can save the path in the Save_path parameter, you can use None without saving
    x_aligned, prob = mtcnn(x, return_prob=True, save_path=path + '/{}.jpg'.format(i))
    i = i + 1
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()  # Extract the feature vector of all face, each vector is 512
# Two two calculations calculation matrix
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(names)
print(pd.DataFrame(dists, columns=names, index=names))
torch.save(embeddings, 'database.pt')  # Of course, you can also save it in a file.
torch.save(names, 'names.pt')
