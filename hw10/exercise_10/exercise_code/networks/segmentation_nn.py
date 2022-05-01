"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams.update(hparams)
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3t = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4t = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #240*240
        #pool 120*120

        self.conv5 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.ConvTranspose2d(512, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.pool = nn.MaxPool2d(2)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv3t.weight)
        nn.init.kaiming_normal_(self.conv4t.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.conv3t.bias)
        nn.init.zeros_(self.conv4t.bias)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(torch.nn.functional.relu(self.conv3t(x)))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.pool(torch.nn.functional.relu(self.conv4t(x)))
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = torch.nn.functional.relu(self.conv7(x))

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
