
import torch
import torchvision.models as models
import torch.nn as nn

class Ensamble_Model(nn.Module):
    def __init__(self, transfer_learning=True, num_classes=10):
        super(Ensamble_Model, self).__init__()
        
        if transfer_learning:
            self.resnet = models.resnet18(pretrained=True)
            for param in self.resnet.parameters():
                param.requires_grad = False
            num_in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(in_features=num_in_features, out_features=num_classes)

            self.vgg = models.vgg16_bn(pretrained=True)
            for param in self.vgg.parameters():
                param.requires_grad = False
            num_in_features = self.vgg.classifier[6].in_features
            self.vgg.classifier[6] = nn.Linear(in_features=num_in_features, out_features=num_classes)


            self.densenet = models.densenet121(pretrained=True)
            for param in self.densenet.parameters():
                param.requires_grad = False
            num_in_features = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Linear(in_features=num_in_features, out_features=num_classes)
        else:
            self.resnet = models.resnet18(pretrained=True)
            num_in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(in_features=num_in_features, out_features=num_classes)

            self.vgg = models.vgg16_bn(pretrained=True)
            num_in_features = self.vgg.classifier[6].in_features
            self.vgg.classifier[6] = nn.Linear(in_features=num_in_features, out_features=num_classes)


            self.densenet = models.densenet121(pretrained=True)
            num_in_features = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Linear(in_features=num_in_features, out_features=num_classes)


    def forward(self, input):
        # predict = self.resnet(input)
        # predict = self.vgg(input)
        predict = self.densenet(input)

        return predict



if __name__ == '__main__':
  model = Ensamble_Model()
  print(model)