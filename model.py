import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class AgePredictionModel(nn.Module):
    def __init__(self, backbone='convnextv2_tiny', task='regression'):
        super(AgePredictionModel, self).__init__()
        self.task = task

        if backbone == 'convnextv2_tiny':
            from timm.models.convnext import convnextv2_tiny
            self.backbone = convnextv2_tiny(pretrained=True, num_classes=0)
            features = self.backbone.num_features
        elif backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=True)
            features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if self.task == 'regression':
            self.head = nn.Linear(features, 1)
        elif self.task == 'group_classification':
            self.head = nn.Linear(features, 5)
        else:
            raise ValueError("task must be 'regression' or 'group_classification'")

        self._init_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def _init_weights(self):
        from timm.models.layers import trunc_normal_
        trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)



class AgePredictionWithGender(nn.Module):
    def __init__(self, backbone='convnextv2_tiny', age_task='regression'):
        super().__init__()
        self.age_task = age_task

        if backbone == 'convnextv2_tiny':
            from timm.models.convnext import convnextv2_tiny
            self.backbone = convnextv2_tiny(pretrained=True, num_classes=0)
            features = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if self.age_task == 'regression':
            self.age_head = nn.Linear(features, 1)
        elif self.age_task == 'group_classification':
            self.age_head = nn.Linear(features, 5)
        else:
            raise ValueError("age_task must be 'regression' or 'group_classification'")

        self.gender_head = nn.Linear(features, 2)
        self._init_weights()

    def forward(self, x):
        features = self.backbone(x)
        age_output = self.age_head(features)
        gender_output = self.gender_head(features)
        return age_output, gender_output

    def _init_weights(self):
        trunc_normal_(self.age_head.weight, std=.02)
        trunc_normal_(self.gender_head.weight, std=.02)
        nn.init.constant_(self.age_head.bias, 0)
        nn.init.constant_(self.gender_head.bias, 0)

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    # model.load_state_dict(torch.load('convnextv2_base_22k_224_ema.pt'))
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


import torch
import torch.nn as nn
#--------------------------------#
# 从torch官方可以下载resnet50的权重
#--------------------------------#
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}
 
#-----------------------------------------------#
# 此处为定义3*3的卷积，即为指此次卷积的卷积核的大小为3*3
#-----------------------------------------------#
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
#-----------------------------------------------#
# 此处为定义1*1的卷积，即为指此次卷积的卷积核的大小为1*1
#-----------------------------------------------#
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 
#----------------------------------#
# 此为resnet50中标准残差结构的定义
# conv3x3以及conv1x1均在该结构中被定义
#----------------------------------#
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        #--------------------------------------------#
        # 当不指定正则化操作时将会默认进行二维的数据归一化操作
        #--------------------------------------------#
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #---------------------------------------------------#
        # 根据input的planes确定width,width的值为
        # 卷积输出通道以及BatchNorm2d的数值
        # 因为在接下来resnet结构构建的过程中给到的planes的数值不相同
        #---------------------------------------------------#
        width           = int(planes * (base_width / 64.)) * groups
        #-----------------------------------------------#
        # 当步长的值不为1时,self.conv2 and self.downsample
        # 的作用均为对输入进行下采样操作
        # 下面为定义了一系列操作,包括卷积，数据归一化以及relu等
        #-----------------------------------------------#
        self.conv1      = conv1x1(inplanes, width)
        self.bn1        = norm_layer(width)
        self.conv2      = conv3x3(width, width, stride, groups, dilation)
        self.bn2        = norm_layer(width)
        self.conv3      = conv1x1(width, planes * self.expansion)
        self.bn3        = norm_layer(planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride
    #--------------------------------------#
    # 定义resnet50中的标准残差结构的前向传播函数
    #--------------------------------------#
    def forward(self, x):
        identity = x
        #-------------------------------------------------------------------------#
        # conv1*1->bn1->relu 先进行一次1*1的卷积之后进行数据归一化操作最后过relu增加非线性因素
        # conv3*3->bn2->relu 先进行一次3*3的卷积之后进行数据归一化操作最后过relu增加非线性因素
        # conv1*1->bn3 先进行一次1*1的卷积之后进行数据归一化操作
        #-------------------------------------------------------------------------#
        out      = self.conv1(x)
        out      = self.bn1(out)
        out      = self.relu(out)
 
        out      = self.conv2(out)
        out      = self.bn2(out)
        out      = self.relu(out)
 
        out      = self.conv3(out)
        out      = self.bn3(out)
        #-----------------------------#
        # 若有下采样操作则进行一次下采样操作
        #-----------------------------#
        if self.downsample is not None:
            identity = self.downsample(identity)
        #---------------------------------------------#
        # 首先是将两部分进行add操作,最后过relu来增加非线性因素
        # concat（堆叠）可以看作是通道数的增加
        # add（相加）可以看作是特征图相加，通道数不变
        # add可以看作特殊的concat,并且其计算量相对较小
        #---------------------------------------------#
        out += identity
        out = self.relu(out)
 
        return out
 
#--------------------------------#
# 此为resnet50网络的定义
# input的大小为224*224
# 初始化函数中的block即为上面定义的
# 标准残差结构--Bottleneck
#--------------------------------#
class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
 
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer   = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes    = 64
        self.dilation    = 1
        #---------------------------------------------------------#
        # 使用膨胀率来替代stride,若replace_stride_with_dilation为none
        # 则这个列表中的三个值均为False
        #---------------------------------------------------------#
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        #----------------------------------------------#
        # 若replace_stride_with_dilation这个列表的长度不为3
        # 则会有ValueError
        #----------------------------------------------#
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
 
        self.block       = block
        self.groups      = groups
        self.base_width  = width_per_group
        #-----------------------------------#
        # conv1*1->bn1->relu
        # 224,224,3 -> 112,112,64
        #-----------------------------------#
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1   = norm_layer(self.inplanes)
        self.relu  = nn.ReLU(inplace=True)
        #------------------------------------#
        # 最大池化只会改变特征图像的高度以及
        # 宽度,其通道数并不会发生改变
        # 112,112,64 -> 56,56,64
        #------------------------------------#
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
 
        # 56,56,64   -> 56,56,256
        self.layer1  = self._make_layer(block, 64, layers[0])
 
        # 56,56,256  -> 28,28,512
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
 
        # 28,28,512  -> 14,14,1024
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1])
 
        # 14,14,1024 -> 7,7,2048
        self.layer4  = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2])
        #--------------------------------------------#
        # 自适应的二维平均池化操作,特征图像的高和宽的值均变为1
        # 并且特征图像的通道数将不会发生改变
        # 7,7,2048 -> 1,1,2048
        #--------------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #----------------------------------------#
        # 将目前的特征通道数变成所要求的特征通道数（1000）
        # 2048 -> num_classes
        #----------------------------------------#
        self.fc      = nn.Linear(512 * block.expansion, num_classes)
 
 
        #-------------------------------#
        # 部分权重的初始化操作
        #-------------------------------#
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #-------------------------------#
        # 部分权重的初始化操作
        #-------------------------------#
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
 
    #--------------------------------------#
    # _make_layer这个函数的定义其可以在类的
    # 初始化函数中被调用
    # block即为上面定义的标准残差结构--Bottleneck
    #--------------------------------------#
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer        = self._norm_layer
        downsample        = None
        previous_dilation = self.dilation
        #-----------------------------------#
        # 在函数的定义中dilate的值为False
        # 所以说下面的语句将直接跳过
        #-----------------------------------#
        if dilate:
            self.dilation *= stride
            stride        = 1
        #-----------------------------------------------------------#
        # 如果stride！=1或者self.inplanes != planes * block.expansion
        # 则downsample将有一次1*1的conv以及一次BatchNorm2d
        #-----------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        #-----------------------------------------------#
        # 首先定义一个layers,其为一个列表
        # 卷积块的定义,每一个卷积块可以理解为一个Bottleneck的使用
        #-----------------------------------------------#
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
 
        return nn.Sequential(*layers)
    #------------------------------#
    # resnet50的前向传播函数
    #------------------------------#
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        #--------------------------------------#
        # 按照x的第1个维度拼接（按照列来拼接，横向拼接）
        # 拼接之后,张量的shape为(batch_size,2048)
        #--------------------------------------#
        x = torch.flatten(x, 1)
        #--------------------------------------#
        # 过全连接层来调整特征通道数
        # (batch_size,2048)->(batch_size,1000)
        #--------------------------------------#
        x = self.fc(x)
        return x
 
# 定义网络
class AgeNet(nn.Module):
    def __init__(self):
        super(AgeNet, self).__init__()

        # 定义网络的卷积层和池化层
        self.cnn64 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn128_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.cnn128_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.cnn128_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn256_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cnn256_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cnn256_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn512_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cnn512_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cnn512_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn512_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cnn512_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cnn512_6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),  # 修改这里的维度
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(4096, 101)  # 输出101类

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 前向计算
        x = self.cnn64(x)
        x = self.maxpool1(x)
        x = self.cnn128_1(x)
        x = self.cnn128_2(x)
        x = self.cnn128_3(x)
        x = self.maxpool2(x)
        x = self.cnn256_1(x)
        x = self.cnn256_2(x)
        x = self.cnn256_3(x)
        x = self.maxpool3(x)
        x = self.cnn512_1(x)
        x = self.cnn512_2(x)
        x = self.cnn512_3(x)
        x = self.maxpool4(x)
        x = self.cnn512_4(x)
        x = self.cnn512_5(x)
        x = self.cnn512_6(x)
        x = self.maxpool5(x)

        # 展平操作，准备进入全连接层
        x = x.view(x.size(0), -1)  # 展平成 [batch_size, 512*4*4]

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
class AgePredictionConditionalOnGender(nn.Module):
    def __init__(self, backbone='convnextv2_tiny'):
        super().__init__()
        # 共享主干网络
        if backbone == 'convnextv2_tiny':
            from timm.models.convnext import convnextv2_tiny
            self.backbone = convnextv2_tiny(pretrained=True, num_classes=0)
            features = self.backbone.num_features
        else:
            raise ValueError("Unsupported backbone")

        # 性别预测 head
        self.gender_head = nn.Linear(features, 2)

        # 分性别年龄预测 head（注意是两个分支）
        self.age_head_male = nn.Linear(features, 1)
        self.age_head_female = nn.Linear(features, 1)

    def forward(self, x):
        features = self.backbone(x)
        gender_logits = self.gender_head(features)
        age_pred_male = self.age_head_male(features)
        age_pred_female = self.age_head_female(features)
        return age_pred_male, age_pred_female, gender_logits

