import os
import numpy as np
import torch
from datasets import FontData
from my_models import Ensamble_Model
from torch.utils.data import DataLoader
from torchvision import transforms
from my_utils.expriment import test_model
from my_utils.plots import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score






data_dir = '/content/drive/MyDrive/Font_Recognition/Data/Test'
batch_size = 64
model_name = 'vgg'

data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale (3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 


image_dataset = FontData(data_dir, data_transform, for_data='Test')
                 
dataloader = DataLoader(image_dataset, batch_size=batch_size)
                                            

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(image_dataset)


model = Ensamble_Model()
checkpoint = torch.load(f'{model_name}_checkpoint.tar', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


targets, predictions = test_model(model, dataloader, device)

cnf_matrix = confusion_matrix(targets , predictions)
acc = accuracy_score(targets , predictions)
np.set_printoptions(precision=2)
print(f'The Accuracy of the model on test data is:{acc}')
print(cnf_matrix)


# Plot non-normalized confusion matrix
class_names = [0,1,2,3,4,5,6,7,8,9]
acc = accuracy_score(targets , predictions)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=f'Confusion matrix, Accuracy {round(acc*100, 2)}%')

