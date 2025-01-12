import torch
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from torchvision import transforms
import matplotlib.patches as patches
import matplotlib.pyplot as plt


class LineDataset(torch.utils.data.Dataset): #Klasa służąca do załadowania obrazów testowych i ich etykiet
    def __init__(self, annotation_file, image_dir, transform=None):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
    transforms.ToTensor(),
    
])
        self.annotations = self.load_annotations()

    def __getitem__(self, idx): #Funkcja transformująca obrazy w tensory
        # Get the image name from the annotation (use idx to get image_name key)
        image_name = list(self.annotations.keys())[idx]
        annotation = self.annotations[image_name]

        # Get the image path
        image_path = os.path.join(self.image_dir, annotation['image_name'])
        image = Image.open(image_path).convert("RGB")

        # Get boxes and labels
        boxes = annotation['boxes']
        labels = annotation['labels']

        # Apply transformations (including ToTensor, if provided)
        image = self.transform(image)  # Apply the transform (including ToTensor())

        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels
        }

        return image, target

    def load_annotations(self): #Funkcja przypisująca etykiety (w formie wielokątów) obrazom
        tree = ET.parse(self.annotation_file)
        root = tree.getroot()

        annotations = {}

        # Iterate through all 'image' elements in XML
        for image in root.findall('image'):
            image_name = image.get('name')
            width = int(image.get('width'))
            height = int(image.get('height'))

            # Initialize lists for boxes and labels
            boxes = []
            labels = []

            # Iterate through polygons in the image
            for polygon in image.findall('polygon'):
                label = polygon.get('label')
                points_str = polygon.get('points')
                points = [tuple(map(float, p.split(','))) for p in points_str.split(';')]

                # Compute bounding box from polygon points
                points = np.array(points)
                xmin, ymin = points.min(axis=0)
                xmax, ymax = points.max(axis=0)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)  # Store the label for the object

            # Convert to PyTorch tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor([self.label_map(label) for label in labels], dtype=torch.int64)  # Map labels to integers

            annotations[image_name] = {'boxes': boxes, 'labels': labels, 'image_name': image_name}

        return annotations

    def __len__(self):
        return len(self.annotations)

    def label_map(self, label): #Funkcja zmieniająca etykiety w formie teksowtej na liczby
        """Map label names to integer values"""
        label_map = {'horizontal_line': 1, 'vertical_line': 2}
        return label_map.get(label, 0)  # Default to 0 if the label is unknown

image_dir = "path to your image dir" #Ścieżka do folderu z obrazami tekstowymi
annotation_file = "path to your annotation file" #Ścieżka do pliku z etykietami
transform = transforms.Compose([
    transforms.ToTensor(),
])
# Stworzenie bazy danych
dataset = LineDataset(annotation_file, image_dir, transform)

#Sprawdzenie ilości obrazów
print(f"Total images: {len(dataset)}")

# Sprawdzenie czy na obrazach znajdują się etykiety
for i in range(8):  # Check the first 5 samples
    image, annotations = dataset[i]
    
    # Print out the details of the annotations
    print(f"Image {i}:")
    
    # Check if 'boxes' and 'labels' are present and not empty
    if 'boxes' in annotations and annotations['boxes'].size(0) > 0:
        print(f"  Found {annotations['boxes'].size(0)} boxes")
    else:
        print("  No boxes found")
    
    if 'labels' in annotations and len(annotations['labels']) > 0:
        print(f"  Found {len(annotations['labels'])} labels")
    else:
        print("  No labels found")
    
    # Optionally, print out a sample of the box coordinates and labels
    print(f"  Box coordinates (first box): {annotations['boxes'][0] if annotations['boxes'].size(0) > 0 else 'N/A'}")
    print(f"  Labels (first label): {annotations['labels'][0] if len(annotations['labels']) > 0 else 'N/A'}")

# Przypisanie odpowiedniego narzędzia do trenowania modelu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stworzenie data loadera
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))  # collate_fn is required for object detection

# Zastosowania optymalizatora (Adam)



# Określenie rodzaju stosowanego modelu
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Określenie szybkości uczenia się
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Modify the classifier for 2 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=3)  # 2 classes + background

# Przygotowanie modelu do treningu
model.train()

# Pętla treningowa
num_epochs = 3 # Ilość powtórzeń treningu
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in data_loader:
        # Move targets to the device
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Move images to the device (images are already tensors due to ToTensor())
        images = [image.to(device) for image in images]  # Move image tensors to the device

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)

        # Total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        losses.backward()
        optimizer.step()

        running_loss += losses.item()


    # Update the learning rate
    lr_scheduler.step()

# Zapis stanu modelu do pliku pth w folderze pracy
torch.save(model.state_dict(), 'model.pth')

# Wczytanie modelu i ustawienie go w tryb ewaluacji
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 3
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('model.pth'))  # Replace with the path to your model
model.to(device)
model.eval()



# Wczytanie obrazu do ewaluacji
image_path = 'path to your plot'
image = Image.open(image_path).convert("RGB")

image_tensor = transform(image).unsqueeze(0)

image_tensor = image_tensor.to(device)

with torch.no_grad():  # No need to calculate gradients during inference
    prediction = model(image_tensor)

# Funkcja odfiltrowująca małe etykiety
def filter_small_boxes(outputs, min_box_area=10000):
    for output in outputs:
        boxes = output['boxes']
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Filter boxes smaller than the threshold
        valid_boxes_mask = (areas >= min_box_area)
        output['boxes'] = boxes[valid_boxes_mask]
        output['labels'] = output['labels'][valid_boxes_mask]
        output['scores'] = output['scores'][valid_boxes_mask]

    return outputs

# Odfiltrowanie małych etykiet
prediction = filter_small_boxes(prediction, 10000)

# Ustalenie progu zaufania
threshold = 0.30 
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# Odfiltrowanie wyników poniżej progu zaufania
indices = torch.where(scores > threshold)[0]
boxes = boxes[indices]
labels = labels[indices]

# Ustalenie nazw etykiet
class_names = {1: "horizontal_line", 2: "vertical_line"}

# Stworzenie wykresu, na który nakładane będą etykiety
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

# Stworzenie etykiet
for i in range(len(boxes)):
    box = boxes[i].cpu().numpy()
    label = labels[i].item()
    label_name = class_names.get(label, "Unknown")
    score = scores[indices[i]].item()
    
    # Create a rectangle patch for the box
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Dopisanie wyników do etykiet
    ax.text(
        box[0], box[1] - 10, f"{label_name}: {score:.2f}",
        color='red', fontsize=12, weight='bold',
        bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='red', boxstyle='round,pad=0.5')
    )

# Pokazanie wykresu
plt.axis('off') 
plt.show()
# Zapis wykresu z etykietami w folderze pracy
plt.savefig('nazwa.png',dpi=500)
