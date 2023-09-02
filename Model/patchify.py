import glob
import cv2
from PIL import Image
from tqdm import tqdm
import patchify
import numpy as np
import glob as glob
import os
import cv2

STRIDE = 50
SIZE_WIDTH = 200
SIZE_HEIGHT = 50

def create_patches(input_paths, out_path):

    input = sorted(glob.glob(f'{input_paths}/*.jpg'), key =lambda x: int(x.split('\\')[-1].split('.')[0].split("cropped")[1]))
    print(f"Creating patches for {len(input)} images")

    for image_path in tqdm(input, total=len(input)):
        image = Image.open(image_path)
        image_name = image_path.split(os.path.sep)[-1].split('.')[0]
        
        # Create patches of size (SIZE_HEIGHT, SIZE_WIDTH, 3)
        patches = patchify.patchify(np.array(image), (SIZE_HEIGHT, SIZE_WIDTH, 3), STRIDE)

        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0, :, :, :]
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"{out_path}/{image_name}_{counter}.png",
                    patch
                )