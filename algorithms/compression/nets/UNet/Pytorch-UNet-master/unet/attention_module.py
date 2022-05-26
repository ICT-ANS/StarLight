import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.upsample_mode = 'bilinear'
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        gl = self.W_g(g)
        gl_size = gl.size()
        xl = self.W_x(x)
        xl_size =xl.size()

        xl = F.interpolate(xl, size=gl_size[2:], mode=self.upsample_mode)
        # concat + relu
        psi = self.relu(gl + xl)

        psi = self.psi(psi)
        psi_resample = F.interpolate(psi, size= xl_size[2:], mode=self.upsample_mode)
        return x * psi_resample

