import numpy as np
import matplotlib.pyplot as plt
import torch

def my_imshow(dl, nrows, ncols, class_names):
    
    fig , axs = plt.subplots(nrows=nrows, ncols=ncols, figsize= (15,10))
    axs = axs.ravel()

    i=0
    
    for image, label in dl:
        
        image = image.numpy()
        image = image.squeeze()
        axs[i].imshow(image, cmap='gray')
        axs[i].set_title(class_names[label.numpy()[0]])
        axs[i].axis('off')
        
        i+=1
        
        if i==nrows*ncols :
            return

def show_misclass(model, dataloader, class_names, device, nrows, ncols):
    model.eval()
    model.to(device)
    images_so_far = 0
        
    fig , axs = plt.subplots(nrows=nrows, ncols=ncols, figsize= (15,10))
    plt.subplots_adjust(hspace = 0.3)
    axs = axs.ravel()
    
    for inputs, labels in dataloader:
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in torch.nonzero(preds != labels):
                image = inputs[i.item()].cpu()
                image = image.numpy()[0,:,:]
                image = image.squeeze()
                
            
                axs[images_so_far].imshow(image, cmap='gray')
                axs[images_so_far].set_title(f"Pred:{class_names[preds[i].cpu().numpy()[0]]} \n (True:{class_names[labels[i].cpu().numpy()[0]]})" )
                axs[images_so_far].axis('off')
                images_so_far+=1

            if images_so_far==nrows*ncols :
                fig.savefig(f'MissClassified_images.png')
                return


