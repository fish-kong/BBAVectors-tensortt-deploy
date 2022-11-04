import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet
from . import mobilenet

import torch.nn.functional as F
from decoder import DecDecoder
class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv,export=False):
        super(CTRBOX, self).__init__()



        assert down_ratio in [2, 4, 8, 16]
        self.export=export
        self.l1 = int(np.log2(down_ratio))
        # channels = [3, 64, 256, 512, 1024, 2048]  # 当下面采用resnet101的时候用这个
        # self.base_network = resnet.resnet101(pretrained=pretrained)
        # self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        # self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        # self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)

        # channels = [3, 64, 64, 512, 1024, 2048]  # 用resnet18的时候是这个
        # self.base_network = resnet.resnet18(pretrained=pretrained)
        # self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
        # self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
        # self.dec_c4 = CombinationModule(512, 256, batch_norm=True)

        channels = [3, 64, 24, 512, 1024, 2048]  # mobilenetv2的时候是这个
        self.base_network = mobilenet.Backbone(pretrained=pretrained)
        self.dec_c2 = CombinationModule(32, 24, batch_norm=True)
        self.dec_c3 = CombinationModule(96, 32, batch_norm=True)
        self.dec_c4 = CombinationModule(320, 96, batch_norm=True)


        if self.export:
            self.decoder = DecDecoder(K=500,
                                         conf_thresh=0.1,
                                         num_classes=heads['hm'])
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   # nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   # nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        #print(c2_combine.size())
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        if self.export:
            result=self.decoder.ctdet_decode(dec_dict,True)
            return result
        # for head in self.heads:
        #     print(head,dec_dict[head].size())
        return dec_dict
if __name__ == '__main__':
    import numpy as np

    heads = {'hm': 3,
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    device='cpu'
    model=CTRBOX(heads, pretrained=False, down_ratio=4, final_kernel=1, head_conv=256,export=False).to(device)
    input=np.ones((1,3,512,512)).astype(np.float32)

    dummy_input = torch.from_numpy(input).to(device)
    logit=model(dummy_input)
    print(logit['hm'].size())
    print(logit['wh'].size())
    print(logit['reg'].size())
    print(logit['cls_theta'].size())

    model=CTRBOX(heads, pretrained=False, down_ratio=4, final_kernel=1, head_conv=256,export=True).to(device)
    logit = model(dummy_input)
    print(logit.size())