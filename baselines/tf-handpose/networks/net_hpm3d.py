import torch
from torch import nn

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

class Hpm3d(nn.Module):
    def __init__(self, num_joints, input_nc):
        super(Hpm3d, self).__init__()
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
        self.depth = Repeat(num_joints)
        self.depth_fc_1 = nn.Linear(num_joints*32*32, 512)
        self.depth_fc_2 = nn.Linear(512, 512)
        self.depth_fc_3 = nn.Linear(512, num_joints)

        self.upsampler = nn.functional.interpolate

    # self.upsampler = nn.Upsample(scale_factor = 8, mode = 'bilinear', align_corners = True)

    def forward(self, x):
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
        out_depth = self.depth(out_6)

        out_depth = out_depth.view(out_depth.shape[0], -1)

        out_depth = self.depth_fc_1(out_depth)
        out_depth = self.depth_fc_2(out_depth)
        out_depth = self.depth_fc_3(out_depth)
        return out_depth

if __name__ == "__main__":
    net = Hpm3d(21, 1)
    input = torch.rand(10, 1, 128, 128)
    out = net(input)
    print(out.shape)
