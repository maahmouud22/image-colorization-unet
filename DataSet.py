import os
import cv2
import torch
from torch.utils.data import Dataset
from skimage.color import rgb2lab
import numpy as np
import random # ⬅️ إضافة المكتبة المطلوبة لعملية الـ Augmentation

class ColorizationDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = []
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(os.path.join(root, file))
        
        self.image_files = self.image_files
        
        if len(self.image_files) == 0:
            print(f"❌ خطأ: لم يتم العثور على أي صور في المسار: {image_dir}")
        else:
            print(f"✅ تم العثور على {len(self.image_files)} صورة بنجاح.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        image = cv2.imread(img_path)
        
        if image is None:
            img_array = np.fromfile(img_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            return self.__getitem__((idx + 1) % len(self.image_files))

        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128))
            
            # ⬅️ تقنية Data Augmentation: قلب الصورة أفقياً بنسبة 50%
            # هذا يجعل الموديل يتعلم على صور تبدو مختلفة، ويمنع الـ Overfitting
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
            
            img_lab = rgb2lab(image).astype("float32")
            L = img_lab[:, :, 0] / 50.0 - 1.0
            ab = img_lab[:, :, 1:] / 110.0
            
            L = torch.from_numpy(L).unsqueeze(0)
            ab = torch.from_numpy(ab).permute(2, 0, 1)
            
            return L, ab
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.image_files))