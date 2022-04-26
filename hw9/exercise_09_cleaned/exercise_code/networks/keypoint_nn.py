"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        #super(KeypointModel, self).__init__()
        super().__init__()
        self.hparams.update(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shap                           #
        # Some layers you might consider including:  e (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        self.conv1 = nn.Conv2d(1, self.hparams['channel_1'], kernel_size=5, padding=2)
        #maxpool -> 96/2=48
        self.conv2 = nn.Conv2d(self.hparams['channel_1'], self.hparams['channel_2'], kernel_size=3, padding=1)
        #maxpool -> 48/2=24

        self.pool = nn.MaxPool2d(2)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

        self.fc = nn.Linear(self.hparams['channel_2']*24*24, 30)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
