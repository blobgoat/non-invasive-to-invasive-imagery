import os
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import pandas as pd

class PatientDicomDataset(Dataset):
    def __init__(self, root_dir, csv_path, transform=None):
        #print("Starting dataset load")
        self.transform = transform
        self.patient_images = []
        self.patient_ids = []
        self.gene_values = []
        print("Loading CSV and images...")
        # Load CSV
        df = pd.read_csv(csv_path, dtype=str)
        df.set_index(df.columns[0], inplace=True)  # Set first column (genes) as index
        df = df.transpose()  # Now rows are patients, columns are genes
        df.index.name = 'patient_id'
        self.corresponding_genes = df.columns.tolist()
        
        # print("CSV loaded successfully.")
        # print("Number of patients in CSV:", len(df))
        # Collect images per patient
        for patient_id in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_id)
           
            if not os.path.isdir(patient_path):
                continue
            #add all images in the patient folder
            #gonna do away with metadata for positionality and such for now
            images = []
            for body_part_folder in os.listdir(patient_path):
                body_part_path = os.path.join(patient_path, body_part_folder)
                if not os.path.isdir(body_part_path):
                    continue
                for image_folder in os.listdir(body_part_path):
                    image_path = os.path.join(body_part_path, image_folder)
                    if not os.path.isdir(image_path):
                        continue

                    for file in os.listdir(image_path):
                        if file.endswith('.dcm'):
                            full_path = os.path.join(image_path, file)
                            images.append(full_path)

            if patient_id in df.index:
                self.patient_images.append(images)
                self.patient_ids.append(patient_id)
                self.gene_values.append(df.loc[patient_id])
                #want to get the first row minus the first column

    def __len__(self):
        return len(self.patient_images)

    def __getitem__(self, idx):
        image_paths = self.patient_images[idx]
        patient_id = self.patient_ids[idx]
        gene_values = self.gene_values[idx]
        corresponding_genes = self.corresponding_genes


        images = []
        for path in image_paths:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            img -= img.min()
            if img.max() != 0:
                img /= img.max()
            img = np.expand_dims(img, axis=0)
            img_tensor = torch.tensor(img)
            if self.transform:
                img_tensor = self.transform(img_tensor)
            images.append(img_tensor)

        return images, patient_id, gene_values, corresponding_genes

def collate_fn(batch):
    images, _, gene_values, _ = zip(*batch)

    # Pad images to match max number of images in batch
    max_images = max(len(img_list) for img_list in images)
    padded_images = []
    for img_list in images:
        pad_count = max_images - len(img_list)
        if pad_count > 0:
            pad = [torch.zeros_like(img_list[0]) for _ in range(pad_count)]
            img_list += pad
        padded_images.append(torch.stack(img_list))

    batch_images = torch.stack(padded_images)  # [B, N, 1, H, W]
    gene_tensor = torch.stack(gene_values)     # [B, G]

    return batch_images, None, gene_tensor, None
