import os
import torch
from datasets import FontData
from my_models import Ensamble_Model
from torch.utils.data import DataLoader
from my_utils.expriment import train_model
from my_utils.expriment import test_model
from torchvision import transforms,datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


data_dir = '/content/drive/MyDrive/Font_Recognition/Data'
batch_size = 128

data_transforms = {
    'Train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale (3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Validation': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale (3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: FontData(os.path.join(data_dir, x),
                                          data_transforms[x], for_data=x)
                  for x in ['Train','Validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True if x== 'Train' else False, num_workers=0)
              for x in ['Train','Validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['Train','Validation']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



print(image_datasets['Train'])
print(image_datasets['Validation'])


model = Ensamble_Model(transfer_learning=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, dataset_sizes, model_name='dense', num_epochs=10)
torch.save(model.state_dict(), 'dense_best_model.pth')

