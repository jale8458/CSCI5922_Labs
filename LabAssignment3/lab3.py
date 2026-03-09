import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image

import os
import json
from typing import Callable
import matplotlib.pyplot as plt
from skimage import io

########## VizWizLoader ##########

class VizWizLoader(torch.utils.data.Dataset):
    def __init__(self, strFolder: str, strAnnotationPath: str, fDataPercentage: float = 1.0,
                 tTransform: Callable[[torch.tensor], torch.tensor] = None) -> None:
        '''
        strFolder: path to unzip'd folder of VizWiz images
        strLabelPath: path to .json file containing the annotations
        fDataPercentage: percentage of available samples to use. Must be normalized between 0.0 and 1.0. Default: 1.0
        tTransform: optional place to connect PyTorch image transformations. Default: converts images to 3x224x224 tensors
        For the train and val splits, returns tuples of the form:
            (image, question text, binary label, answer texts)
        Otherwise, returns tuples of the form:
            (image, question text)
        '''
        self.strFolder = strFolder
        if self.strFolder[-1] != "/": self.strFolder += "/"
        self.tTransform = tTransform
        vecPaths = os.listdir(self.strFolder)
        self.strPrefix = vecPaths[0].split("_")[1]

        with open(strAnnotationPath, "r") as f:
            self.vecAnnos = json.load(f)
        
        self.iN = int(fDataPercentage * len(self.vecAnnos))

        if self.tTransform is None:
            self.tTransform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True), v2.RandomCrop(224)])
            self.tTransformUndersized = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True), v2.Resize((224, 224))])

        return

    def __len__(self) -> int: return self.iN

    def __getitem__(self, idx: int) -> tuple:
        if idx > self.iN:
            print(f"Error! Tried to access index {idx} but only {self.iN} samples are available")
            return None
        
        strPath = self.strFolder + self.vecAnnos[idx]["image"]
        imX = Image.open(strPath)
        w, h = imX.size
        if w >= 224 and h >= 224: tX = self.tTransform(imX)
        else: tX = self.tTransformUndersized(imX)

        if self.strPrefix == "test":
            return tX, self.vecAnnos[idx]["question"]
        else:
            return tX, self.vecAnnos[idx]["question"], self.vecAnnos[idx]["answerable"], self.vecAnnos[idx]["answers"]
        
########## Testing/Debugging Functions ##########
# Plot image from the image url
def visualize_image(image_url):
    image = io.imread(image_url)
    print(image_url)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def findMeanStd(dataset):
    # Finds mean and std of dataset and returns then as a list
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    # Compute per-channel mean/std over the whole training set
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    n_pixels = 0
    # Compute batch by batch to prevent runtime crash
    for x, *_ in train_loader:
        sum_ += x.sum(dim=(0, 2, 3))
        sum_sq += (x ** 2).sum(dim=(0, 2, 3))
        n_pixels += x.size(0) * x.size(2) * x.size(3)

    # Print results
    mean = sum_ / n_pixels
    std = (sum_sq / n_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()
    
######### Main ##########

# # Directory for all images
# img_dir = 'https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/'

# # Directory for annotation files
# ann_dir = 'https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations/'

def main():

    # Use GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    ##### Parameters #####
    dataPercent = 0.49
    batchSize = 16

    ##### Preprocessing #####

    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        # Precalculated mean and std
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
        # # To find mean and std deviation
        # v2.Normalize(findMeanStd(VizWizLoader("./data/train/images", "./data/train/annotations", dataPercent)))
    ])
    train_set = VizWizLoader("./data/train/images", "./data/train/annotations", dataPercent, transform)
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)


if __name__ == "__main__":
    main()