import torch
from torchvision import models
import torch.nn as nn
#from torchvision.models.feature_extraction import create_feature_extractor

class _CutPasteNetBase(nn.Module):
    # forward outputs: logits
    def __init__(self, encoder = 'resnet18', pretrained = True, dims = [512, 512,128], num_class = 3):
        super().__init__()
        self.encoder = getattr(models, encoder)(pretrained = pretrained)
        #print("encoder: ",self.encoder)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        last_layer= list(self.encoder.named_modules())[-1][0].split('.')[0]
        setattr(self.encoder, last_layer, nn.Identity())
        proj_layers = []
        for d in dims[:-1]:
            proj_layers.append(nn.Linear(d, d, bias=False)),
            proj_layers.append((nn.BatchNorm1d(d))),
            proj_layers.append(nn.ReLU(inplace=True))
        embeds = nn.Linear(dims[-2], dims[-1], bias=num_class > 0)
        proj_layers.append(embeds)
        self.head = nn.Sequential(
            *proj_layers
        )
        self.out = nn.Linear(dims[-1], num_class)
        #print("proj_layers: ",proj_layers)

    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return logits

    def freeze(self, layer_name):
        #freeze encoder until layer_name
        check = False
        for name, param in self.encoder.named_parameters():
            if name == layer_name:
                check = True 
            if not check and param.requires_grad != False:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    """ def create_graph_model(self,):
        return create_feature_extractor(model=self, return_nodes=["head", "out"]) """
    
    
class CutPasteNet(_CutPasteNetBase):
    # forward outputs:  (logits, embeds)
    def __init__(self, encoder='resnet18', pretrained=False, dims=[512, 512, 512, 128], num_class=3):
        super().__init__(encoder, pretrained, dims, num_class)
        return
    
    def forward(self, x):
        features = self.encoder(x)
        #print('feat: ', features.size())
        embeds = self.head(features)
        #print('embed: ', embeds.size())
        logits = self.out(embeds)
        #print('logits: ', logits.size())
        return logits

if __name__=="__main__":
    cutPasteNet = CutPasteNet()
    x = torch.rand((3,1,96,96))     #batch size = 1
    out = cutPasteNet(x)
