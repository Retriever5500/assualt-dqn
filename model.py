import torch
import torch.nn as nn

# TODO
class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()

        # images should be preprocessed (extract luminance channel from RGB channels) by φ defined in the paper.

        # in: (4, 84, 84) - out: (32, 20, 20)
        # reLU should be applied on the outputs
        self.conv1 = nn.Conv2d(inchannels=4, out_channels=32, kernel_size= (8, 8), stride=4) 

        # in: (32, 20, 20) - out: (64, 9, 9)
        # reLU should be applied on the outputs
        self.conv2 = nn.Conv2d(inchannels=32, out_channels=64, kernel_size= (4, 4), stride=2)

        # in: (64, 9, 9) - out: (64, 7, 7)
        # reLU should be applied on the outputs
        self.conv3 = nn.Conv2d(inchannels=64, out_channels=64, kernel_size= (3, 3), stride=1)

        # flattening should be applied here before feeding into fc1

        # in: 64 * 7 * 7 = (3136, ) - out: (512, ) 
        # reLU should be applied on the outputs
        self.fc1 = nn.Linear(in_features=3136, out_features=512)

        # in: (512, ) - out: (7, )
        self.fc2 = nn.Linear(in_features=512, out_features=7)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(x)
         
        x = self.conv2(x)
        x = nn.ReLU(x)
         
        x = self.conv3(x)
        x = nn.ReLU(x)

        x = nn.Flatten(start_dim=1, end_dim=-1)

        x = self.fc1(x)
        x = nn.ReLU(x)

        return self.fc2(x)