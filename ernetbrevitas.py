# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Minahil Raza


from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
import torch


total_bits = 8   #width for weights and activations
n = 7
debug= False
class ACFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ''' 
        Dilated Convolution

        i = input
        o = output
        p = padding
        k = kernel_size
        s = stride
        d = dilation
        
        o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        '''

        self.conv1 = qnn.QuantConv2d(in_channels= in_channels,
                                     out_channels= in_channels,
                                     kernel_size= 3,
                                     padding= 0,
                                     bias= False,
                                     dilation=1,
                                     groups=in_channels,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width= 4,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        self.conv2 = qnn.QuantConv2d(in_channels= in_channels,
                                     out_channels= in_channels,
                                     kernel_size= 3,
                                     padding= 1,
                                     bias= False,
                                     dilation=2,
                                     groups=in_channels,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width= 4,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        self.conv3 = qnn.QuantConv2d(in_channels= in_channels,
                                     out_channels= in_channels,
                                     kernel_size= 3,
                                     padding= 2,
                                     bias= False,
                                     dilation=3,
                                     groups=in_channels,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width= 4,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        self.fused_conv = qnn.QuantConv2d(in_channels= in_channels*3,
                                     out_channels= out_channels,
                                     kernel_size= 1,
                                     padding= 0,
                                     bias= False,
                                     dilation=1,
                                     groups=1,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width= 4,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        if debug:
            print('Shape of input in ACFF Forward= ', x.shape)
            print('Output of layer1(x): ', self.conv1(x).shape)
            print('Output of layer2(x): ', self.conv2(x).shape)
            print('Output of layer3(x): ', self.conv3(x).shape)

        # Fusion
        out = torch.cat((self.conv1(x), self.conv2(x), self.conv3(x)), 1)

        if debug:
            print('Shape after concat in ACFF forward: ', out.shape)

        out = self.fused_conv(out)
        out = self.leaky_relu(out)
        out = self.batch_norm(out)
        out = self.dropout(out)

        if debug:
            print('Final shape of ACFF out: ', out.shape, '\n')

        return out



class ErNET(nn.Module):
    def __init__(self):
        super(ErNET, self).__init__()

        self.conv1 = qnn.QuantConv2d(in_channels= 3,
                                     out_channels= 16,
                                     kernel_size= 3,
                                     padding= 0,
                                     bias= False,
                                     stride=2,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width= 16,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)

        self.acff1 = ACFF(16, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff2 = ACFF(64, 96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff3 = ACFF(96, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff4 = ACFF(128, 128)
        self.acff5 = ACFF(128, 128)
        self.acff6 = ACFF(128, 256)
        self.conv2=qnn.QuantConv2d(in_channels= 256,
                                     out_channels= 5,
                                     kernel_size= 1,
                                     padding= 0,
                                     stride=1,
                                     bias= False,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width= 16,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        self.globalpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
        self.fc= qnn.QuantLinear(3*3*5, 5,
                                     bias= True,
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=16,
                                     weight_restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                     weight_scaling_impl_type=ScalingImplType.CONST,
                                     weight_scaling_const=1.0)
        
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.acff1(out)
        out = self.pool1(out)
        out = self.acff2(out)
        out = self.pool2(out)
        out = self.acff3(out)
        out = self.pool3(out)
        out = self.acff4(out)
        out = self.acff5(out)
        out = self.acff6(out)
        out = self.conv2(out)
        out = self.globalpool(out)

        if debug:
            print('Shape of globalpool output: ', out.shape)

        out = out.view(-1, 5 * 3 * 3)
        out = self.fc(out)
        out = self.soft(out)

        if debug:
            print('Final shape of ErNET Output: ', out.shape)

        return out


          #fractional part
