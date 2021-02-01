import torch
import torch.nn as nn
import torch.nn.functional as F
import efficientnet

# image = torch.randn(8, 3, 128, 128)
# model = efficientnet.efficientnet(net="B3", pretrained=True)
# features=model.features(image)
# print(features.shape)

model2 = efficientnet.efficientnet(net="B3", pretrained=True).to('cuda')



class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out




class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=True, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.new = VGGBlock(512,256,1)
        self.new1 = VGGBlock(256,128,1)
        self.new2 = VGGBlock(128,64,1)
        self.new3 = VGGBlock(64,32,1)
        self.new4 = VGGBlock(40,64,64)
        self.new5 = VGGBlock(32,128,128)
        self.new6 = VGGBlock(48,256,256)
        self.new7 = VGGBlock(96,512,512)
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])


        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)



    def forward(self, input):
        # x0_0 = self.conv0_0(input)
        x0 = model2.features[0:0](input)
        x0_0=self.conv0_0(x0)
        # print(x0_0.shape,"00")

        # x1_0 = self.conv1_0(self.pool(x0_0))

        x1 = model2.features[0:1](input)
        x1_0 = self.new4(x1)
        # print(x1_0.shape,"10")
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))


        # x2_0 = self.conv2_0(self.pool(x1_0))
        x2 = model2.features[0:5](input)
        x2_0 = self.new5(x2)
        # print(x2_0.shape,"20")

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        # x3_0 = self.conv3_0(self.pool(x2_0))
        x3 = model2.features[0:9](input)
        x3_0 = self.new6(x3)

        # print(x3_0.shape,"30")

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # x4_0 = self.conv4_0(self.pool(x3_0))
        x4 = model2.features[0:13](input)
        x4_0 =self.new7(x4)

        # print(x4_0.shape,"40")

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print(x4_0.shape,"a")
        # print(self.up1(x4_0).shape,"aa")

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            # t1 = self.new(self.up1(x4_0))
            # t2 = self.new1(self.up2(x3_1))
            # t3 = self.new2(self.up3(x2_2))
            # t4 = self.new3(self.up(x1_3))
            # tfinal = torch.cat([t1,t2,t3,t4,x0_4],1)
            # y=torch.mean(tfinal,1, keepdim=True)

             # t = self.conv0_1( self.up(x1_0))


            y = torch.cat([output1, output2, output3, output4],1)
            # # print(y.shape)
            y=torch.mean(y,1, keepdim=True)
            # print(y.shape)
            return y
        else:
            output = self.final(x0_4)
            return output

# model = NestedUNet(1,3)
# y = torch.rand([8,3,128,128])
# model(y)


