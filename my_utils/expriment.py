import os
import torch
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, dataset_sizes, model_name='EnsambleModel', num_epochs=25):
    if torch.cuda.is_available():
        print('--------- Training in Cuda Mode ----------')
    else:
        print('--------- Training in Cpu Mode ----------')

    if os.path.exists(f'{model_name}_checkpoint.tar'):
        checkpoint = torch.load(f'{model_name}_checkpoint.tar', map_location={'cuda:1':'cuda:0'})
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        last_epoch = checkpoint['epoch']
        last_time_elapsed = checkpoint['time']
        best_acc = checkpoint['best_acc']
        losses= checkpoint['losses']
    else:
        last_epoch = 0
        last_time_elapsed = 0
        best_acc = 0.0
        losses= {'Train':[], 'Validation':[]}

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    
    model = model.to(device)
    t = tqdm(total=len(dataloaders['Train']))
    for epoch in range(last_epoch, num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs}')
        print('-' * 10)

        t.refresh()
        for phase in ['Train', 'Validation']:
            running_corrects = 0
            running_loss = 0
            if phase == 'Train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval() 
            
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == 'Train':
                    loss.backward()
                    optimizer.step()
                    t.update(1)



                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # t.update(1)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # save checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': best_model_wts,
                    'best_acc' : best_acc,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'time': time.time() - since,
                    'losses' : losses
                   },f'{model_name}_checkpoint.tar')

    
    time_elapsed = time.time() - since + last_time_elapsed
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    plt.plot(losses['Train'], label='Training loss')
    plt.plot(losses['Validation'], label='Validation loss')
    idx = np.argmin(losses['Validation'])
    plt.plot([idx,idx], [0,losses['Validation'][idx]],'--', color= 'coral' , label = 'Best' )
    plt.plot([0,idx], [losses['Validation'][idx],losses['Validation'][idx]],'--', color= 'coral' )
    plt.plot(idx, losses['Validation'][idx] , 'o' ,color = 'coral', markersize = 8, markerfacecolor = "None" )
    plt.legend()
    plt.ylabel('Loss', fontsize= 12)
    plt.xlabel('Epoch',fontsize= 12)
    plt.tight_layout()
    plt.title('EnsembleModel Train-Validation loss ',fontsize= 12)
    plt.savefig(f'loss_{model_name}.png',bbox_inches='tight')
    plt.show()
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




def test_model(model, dataloader, device):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('--------- Cuda Mode ----------')
    else:
        print('--------- Cpu Mode ----------')
    print('Predicting...')
    predictions = np.array([])
    targets = np.array([])
    model.to(device)
    model.eval()   # Set model to evaluate mode
    since = time.time()
    # Iterate over data.
    t = tqdm(total=len(dataloader))
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        predictions =  np.append(predictions,preds.cpu().numpy())
        targets =  np.append(targets,labels.cpu().numpy())
        t.update(1)

    time_elapsed = time.time() - since
    print('\nTesting completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return targets, predictions


        