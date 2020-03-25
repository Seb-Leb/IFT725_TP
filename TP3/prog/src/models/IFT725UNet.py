# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel

from models.CNNBlocks import ResidualBlock, DenseBlock

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau IFT725UNet.  Un réseau inspiré de UNet
mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

'''

class IFT725UNet(CNNBaseModel):
    """
     Class that implements a brand new segmentation CNN
    """

    def __init__(self, num_classes=4, init_weights=True, dense_encode=False, dense_decode=False, residual_encode=False,
                 residual_decode=False):
        """
        Builds IFT725UNet model.
        Args:
            num_classes(int): number of classes. default 4(acdc)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
            dense_encode(bool): when true dense blocks will be used in the encode. default False
            dense_decode(bool): when true dense blocks will be used in the decode. default False
            residual_encode(bool): when true residual blocks will be used in the encode. default False
            residual_decode(bool): when true residual blocks will be used in the decode. default False
        """
        super(IFT725UNet, self).__init__(num_classes, init_weights)

        # define hyper-parameters
        self.dense_encode = dense_encode
        self.dense_decode = dense_decode
        self.residual_encode = residual_encode
        self.residual_decode = residual_decode

        # Encode
        in_channels = 1  # gray image
        self.conv_encoder1 = self._contracting_block(in_channels=in_channels, out_channels=64)
        self.max_pool_encoder1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder2 = self._contracting_block(64, 128)
        self.max_pool_encoder2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder3 = self._contracting_block(128, 256)
        self.max_pool_encoder3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder4 = self._contracting_block(256, 512)
        self.max_pool_encoder4 = nn.MaxPool2d(kernel_size=2)

        # Transitional block
        self.transitional_block = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        # Decode
        self.conv_decoder4 = self._expansive_block(1024, 512, 256)
        self.conv_decoder3 = self._expansive_block(512, 256, 128)
        self.conv_decoder2 = self._expansive_block(256, 128, 64)
        self.final_layer = self._final_block(128, 64, num_classes)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # Encode
        encode_block1 = self.conv_encoder1(x)
        encode_pool1 = self.max_pool_encoder1(encode_block1)
        encode_block2 = self.conv_encoder2(encode_pool1)
        encode_pool2 = self.max_pool_encoder2(encode_block2)
        encode_block3 = self.conv_encoder3(encode_pool2)
        encode_pool3 = self.max_pool_encoder3(encode_block3)
        encode_block4 = self.conv_encoder4(encode_pool3)
        encode_pool4 = self.max_pool_encoder4(encode_block4)

        # Transitional block
        middle_block = self.transitional_block(encode_pool4)

        # Decode
        decode_block4 = torch.cat((middle_block, encode_block4), 1)
        cat_layer3 = self.conv_decoder4(decode_block4)
        decode_block3 = torch.cat((cat_layer3, encode_block3), 1)
        cat_layer2 = self.conv_decoder3(decode_block3)
        decode_block2 = torch.cat((cat_layer2, encode_block2), 1)
        cat_layer1 = self.conv_decoder2(decode_block2)
        decode_block1 = torch.cat((cat_layer1, encode_block1), 1)
        final_layer = self.final_layer(decode_block1)
        return final_layer

    def _contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        Building block of the contracting part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            DenseBlock(out_channels, out_channels) if self.dense_encode else nn.Sequential(),
            ResidualBlock(out_channels, out_channels) if self.residual_encode else nn.Sequential()
        )
        return block

    def _expansive_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Building block of the expansive part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            DenseBlock(out_channels, out_channels) if self.dense_decode else nn.Sequential(),
            ResidualBlock(out_channels, out_channels) if self.residual_decode else nn.Sequential()
        )
        return block

    @staticmethod
    def _final_block(in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Final block of the UNet model
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

'''
Fin de votre code.
'''