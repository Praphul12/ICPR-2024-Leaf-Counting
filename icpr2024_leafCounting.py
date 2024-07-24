import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for soil and similar colors (cream, brown, etc.)
    lower_soil = np.array([0, 30, 30])
    upper_soil = np.array([30, 255, 255])
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])
    lower_cream = np.array([0, 0, 180])
    upper_cream = np.array([25, 50, 255])
    
    # Combine the masks for soil, brown, and cream colors
    soil_mask = cv2.inRange(hsv_image, lower_soil, upper_soil)
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
    cream_mask = cv2.inRange(hsv_image, lower_cream, upper_cream)
    combined_mask = soil_mask | brown_mask | cream_mask
    
    # Invert the mask to keep green and similar colors
    inverted_mask = cv2.bitwise_not(combined_mask)
    
    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=inverted_mask)
    
    # Zoom into the center of the image with a 450 ROI
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    roi_size = 450
    top_left_y = center_y - roi_size // 2
    top_left_x = center_x - roi_size // 2
    zoomed_image = result_image[top_left_y:top_left_y + roi_size, top_left_x:top_left_x + roi_size]
    
    return zoomed_image

class LeafCountDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = preprocess_image(img_path)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Convert image to PyTorch tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, img_name

class LeafCountCNN(nn.Module):
    def __init__(self):
        super(LeafCountCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def inference_testing(model_path, img_dir, output_csv):
    # Create dataset and data loader for all images
    dataset = LeafCountDataset(image_dir=img_dir, transform=None)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the trained model
    model = LeafCountCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inference and evaluation loop
    results = []
    with torch.no_grad():
        for images, image_files in tqdm(dataloader, desc="Processing images"):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.squeeze().cpu().numpy()
            
            for i in range(len(predictions)):
                predicted = int(round(predictions[i]))
                filename = os.path.basename(image_files[i])  # Extract just the filename
                results.append((filename, predicted))
                print(f"Processing Image {filename}: Predicted = {predicted}")
                
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['Filename', 'Predicted Leaf Count'])
    results_df.to_csv(output_csv, index=False)
    print(f"Predicted leaf counts saved to '{output_csv}'")
                
    return results

if __name__ == "__main__":
    # Set the paths for model, images and output CSV file
    model_path = r"C:\ICPR 2024\leaf_count_cnn.pth"
    img_dir = r"C:\ICPR 2024\Test_data"
    output_csv = r"C:\ICPR 2024\predicted_leaf_counts.csv"
    
    # Run the inference testing
    predicted_leaf_counts = inference_testing(model_path, img_dir, output_csv)
