from torch import nn
import torch

class Repeat(nn.Module):
    def __init__(self, num_joints):
        super(Repeat, self).__init__()
        self.conv1 = nn.Conv2d(128 + num_joints, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv7 = nn.Conv2d(128, num_joints, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.relu(self.conv5(out))
        out = self.relu(self.conv6(out))
        out = self.conv7(out)
        return out

class Hpm2d(nn.Module):
    def __init__(self, num_joints, input_nc, isTrain):
        """
        This network is part of DGGAN hand pose estimation module.
        It takes as input a RGB image and output n number of heatmap indicating the location of the joints.
        :param num_joints: number of joints
        :param input_nc: number of input channel.
        :param isTrain: whether to caluclate loss
        """
        super(Hpm2d, self).__init__()
        self.isTrain = isTrain
        # self.criterion = criterion
        self.pool = nn.MaxPool2d(2, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(input_nc, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3_CPM = nn.Conv2d(512, 128, kernel_size=3, padding=1)

        self.conv6_1_CPM = nn.Conv2d(128, 512, kernel_size=1, padding=0)
        self.conv6_2_CPM = nn.Conv2d(512, num_joints, kernel_size=1, padding=0)

        self.stage2 = Repeat(num_joints)
        self.stage3 = Repeat(num_joints)
        self.stage4 = Repeat(num_joints)
        self.stage5 = Repeat(num_joints)
        self.stage6 = Repeat(num_joints)

        self.upsampler = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        # debug
        # print("input shape: {}".format(x.shape))
        # image = x['img'].to(self.device)
        out = self.relu(self.conv1_1(x))
        out = self.relu(self.conv1_2(out))
        out = self.pool(out)

        out = self.relu(self.conv2_1(out))
        out = self.relu(self.conv2_2(out))
        out = self.pool(out)

        out = self.relu(self.conv3_1(out))
        out = self.relu(self.conv3_2(out))
        out = self.relu(self.conv3_3(out))
        out = self.relu(self.conv3_4(out))
        out = self.pool(out)

        out = self.relu(self.conv4_1(out))
        out = self.relu(self.conv4_2(out))
        out = self.relu(self.conv4_3(out))
        out = self.relu(self.conv4_4(out))

        out = self.relu(self.conv5_1(out))
        out = self.relu(self.conv5_2(out))
        out_0 = self.relu(self.conv5_3_CPM(out))

        out_1 = self.relu(self.conv6_1_CPM(out_0))
        out_1 = self.conv6_2_CPM(out_1)

        out_2 = torch.cat((out_1, out_0), 1)
        out_2 = self.stage2(out_2)

        out_3 = torch.cat((out_2, out_0), 1)
        out_3 = self.stage3(out_3)

        out_4 = torch.cat((out_3, out_0), 1)
        out_4 = self.stage4(out_4)

        out_5 = torch.cat((out_4, out_0), 1)
        out_5 = self.stage5(out_5)

        out_6 = torch.cat((out_5, out_0), 1)
        out_6 = self.stage6(out_6)

        # print("out6 shape: {}".format(out_6.shape))
        outputs = [out_1, out_2, out_3, out_4, out_5, out_6]
        outputs = [self.upsampler(out) for out in outputs]
        return outputs

if __name__ == "__main__":
    net = Hpm2d(21, 1, True)
    input = torch.rand(10, 1, 128, 128)
    output = net(input)
    print(output.shape)
