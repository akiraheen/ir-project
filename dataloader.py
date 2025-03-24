import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import glob
from PIL import Image
from torchvision import transforms
import multiprocessing

class ImageMetadataDataset(Dataset):
    def __init__(self, json_dir, image_dir, transform=None):
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.transform = transform

        # Get all JSON files
        self.json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))

        self.data_pairs = []
        for json_file in self.json_files:
            base_name = os.path.basename(json_file).split('.')[0]
            base_name = base_name[4:]

            img_path = os.path.join(image_dir, "img" + base_name + '.jpg')

            if os.path.exists(img_path):
                self.data_pairs.append((json_file, img_path))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        json_path, img_path = self.data_pairs[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # extract fields from metadata
        metadata_out = {
            'totalTime': metadata.get('totalTime', 'Unknown'),
            'ingredientLines': metadata.get('ingredientLines', []),
            'attribution': metadata.get('attribution', 'Unknown'),
            'name': metadata.get('name', 'Unknown'),
            'rating': metadata.get('rating', 0.0),
            'numberOfServings': metadata.get('numberOfServings', 0),
            'yield': metadata.get('yield', 'Unknown'),
            'nutritionEstimates': metadata.get('nutritionEstimates', []),
            'source': metadata.get('source', 'Unknown'),
            'flavors': metadata.get('flavors', 'Unknown'),
            'images': metadata.get('images', []),
            'attributes': metadata.get('attributes', {}),
            'id': metadata.get('id', 'Unknown'),
            'totalTimeInSeconds': metadata.get('totalTimeInSeconds', 0),
            'prepTimeInSeconds': metadata.get('prepTimeInSeconds', 0),
            'cookTimeInSeconds': metadata.get('cookTimeInSeconds', 0),
            'cookTime': metadata.get('cookTime', 'Unknown'),
            'prepTime': metadata.get('prepTime', 'Unknown'),
        }

        return {'image': image, 'metadata': metadata_out}


def custom_collate(batch):

    images = torch.stack([item['image'] for item in batch])
    metadata = [item['metadata'] for item in batch]

    return {'image': images, 'metadata': metadata}


import matplotlib.pyplot as plt


# def visualize_sample(index):
#     sample = dataset[index]
#     image = sample['image'].permute(1, 2, 0).numpy()
#     metadata = sample['metadata']
#
#     plt.imshow(image)
#     plt.axis("off")
#     plt.title(f"Metadata: {metadata}")
#     plt.show()



if __name__ == '__main__':

    multiprocessing.freeze_support()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # add other geometric transformations (rotation, inversion, resizing) here
        # blurry, noisy, low-light instead of rotation, inversion, resizing because these are probably more common while taking photos?
    ])

    dataset = ImageMetadataDataset(json_dir='metadata27638', image_dir='images27638', transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate
    )

    for batch in dataloader:
        images = batch['image']
        metadata = batch['metadata']

