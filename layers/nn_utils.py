import torch
import torch.nn as nn
import torch.nn.functional as F
import layers.senet as senet
import layers.resnet as resnet

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, 
            groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class TUM(nn.Module):
    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes

        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)), BasicConv(self.in1, self.planes, 3, 2, 1))
        for i in range(self.scales-2):
            if not i == self.scales - 3:
                self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 2, 1)
                        )
            else:
                self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 1, 0)
                        )
        self.toplayer = nn.Sequential(BasicConv(self.planes, self.planes, 1, 1, 0))
        
        self.latlayer = nn.Sequential()
        for i in range(self.scales-2):
            self.latlayer.add_module(
                    '{}'.format(len(self.latlayer)),
                    BasicConv(self.planes, self.planes, 3, 1, 1)
                    )
        self.latlayer.add_module('{}'.format(len(self.latlayer)),BasicConv(self.in1, self.planes, 3, 1, 1))

        if self.is_smooth:
            smooth = list()
            for i in range(self.scales-1):
                smooth.append(
                        BasicConv(self.planes, self.planes, 1, 1, 0)
                        )
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='interp'):
        _,_,H,W = y.size()
        if fuse_type=='interp':
            return F.interpolate(x, size=(H,W), mode='nearest') + y
        else:
            raise NotImplementedError
            #return nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x,y],1)
        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)
        
        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                    self._upsample_add(
                        deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers)-1-i])
                        )
                    )
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(
                        self.smooth[i](deconved_feat[i+1])
                        )
            return smoothed_feat
        return deconved_feat

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def get_backbone(backbone_name='vgg16'):

    if backbone_name=='vgg16':
        base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        return vgg(base, 3, batch_norm=False)
    elif backbone_name in senet.__all__:
        return getattr(senet,backbone_name)(num_classes=1000, pretrained='imagenet')
    elif backbone_name in resnet.__all__:
        return getattr(resnet,backbone_name)(pretrained=True)


class SFAM(nn.Module):
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels,
                                                 self.planes*self.num_levels // 16,
                                                 1, 1, 0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels // 16,
                                                 self.planes*self.num_levels,
                                                 1, 1, 0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf*_tmp_f)
        return attention_feat

def check_argu(key, value):
    '''
    Check whether the arguments available for constructing m2det modules
    '''
    if key == 'backbone':
        assert value in ['vgg16','resnet18','resnet34','resnet50','resnet101','resnet152'
          'se_resnet50','se_resnet101', 'senet154', 'se_resnet152', 
          'se_resnext50_32x4d', 'se_resnext101_32x4d'], 'Not implemented yet!' # you can do this yourself

    elif key == 'net_family':
        assert value in ['vgg', 'res'], 'Only support vgg and res family Now'
    elif key == 'base_out':
        assert len(value) == 2, 'We have to ensure that the base feature is formed with 2 backbone features'
    elif key == 'planes':
        pass # No rule for plane now.
    elif key == 'num_levels':
        assert value>1, 'At last, you should leave 2 levels'
    elif key == 'num_scales':
        pass # num_scales should equals to len(step_pattern), len(size_pattern)-1,
    elif key == 'sfam':
        pass
    elif key == 'smooth':
        pass
    elif key == 'num_classes':
        pass
    return True
