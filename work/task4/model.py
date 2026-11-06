import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent


class Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())


class VesselNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.enc = resnet18(weights=None)
        if pretrained:
            path = HERE / 'resnet18-f37072fd.pth'
            self.enc.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.enc.fc = nn.Identity()
        self.stem = Block(3, 16)
        self.up3 = Block(512 + 256, 128)
        self.up2 = Block(128 + 128, 96)
        self.up1 = Block(96 + 64, 64)
        self.up0 = Block(64 + 64, 32)
        self.upfull = Block(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    @staticmethod
    def join(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1)

    def forward(self, x):
        full = self.stem(x)
        h = self.enc.relu(self.enc.bn1(self.enc.conv1(x)))
        a = self.enc.layer1(self.enc.maxpool(h))
        b = self.enc.layer2(a)
        c = self.enc.layer3(b)
        d = self.enc.layer4(c)
        x = self.up3(self.join(d, c))
        x = self.up2(self.join(x, b))
        x = self.up1(self.join(x, a))
        x = self.up0(self.join(x, h))
        return self.head(self.upfull(self.join(x, full)))

    def train(self, mode=True):
        super().train(mode)
        for layer in self.enc.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self


def normalize(x):
    return ((x - x.new_tensor([.485, .456, .406])[None, :, None, None]) /
            x.new_tensor([.229, .224, .225])[None, :, None, None]).contiguous(memory_format=torch.channels_last)


def augment(x, y):
    batch = len(x)
    angle = torch.rand(batch, device=x.device) * (2 * math.pi)
    scale = .85 + .3 * torch.rand(batch, device=x.device)
    flip = torch.where(torch.rand(batch, device=x.device) < .5, -1., 1.)
    theta = torch.zeros((batch, 2, 3), device=x.device)
    c, s = angle.cos() / scale, angle.sin() / scale
    theta[:, 0, 0], theta[:, 0, 1] = c * flip, -s
    theta[:, 1, 0], theta[:, 1, 1] = s * flip, c
    theta[:, :, 2] = (torch.rand(batch, 2, device=x.device) - .5) * .12
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    y = F.grid_sample(y, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    gamma = .8 + .4 * torch.rand((batch, 1, 1, 1), device=x.device)
    gain = .85 + .3 * torch.rand((batch, 1, 1, 1), device=x.device)
    color = .9 + .2 * torch.rand((batch, 3, 1, 1), device=x.device)
    x = (x.clamp(0, 1).pow(gamma) * gain * color).clamp(0, 1)
    return normalize(x), y


def loss_fn(logits, target):
    p = logits.float().sigmoid()
    intersection = (p * target).sum((1, 2, 3))
    dice = (2 * intersection + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits.float(), target) + (1 - dice).mean()


@torch.inference_mode()
def predict(model, images, batch_size=2, transforms=1, amp=False):
    """D4 transforms in fixed order, always restore orientation before averaging."""
    model.eval()
    outputs = []
    choices = [(False, ())]
    if transforms >= 4:
        choices = [(False, ()), (False, (-1,)), (False, (-2,)), (False, (-2, -1))]
    if transforms == 8:
        choices += [(True, f) for _, f in choices.copy()]
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size]
        if x.dtype == torch.uint8:
            x = x.float() / 255
        total = None
        for transposed, flips in choices:
            a = x.transpose(-1, -2) if transposed else x
            if flips:
                a = a.flip(flips)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                p = model(normalize(a)).float().sigmoid()
            if flips:
                p = p.flip(flips)
            if transposed:
                p = p.transpose(-1, -2)
            total = p if total is None else total + p
        outputs.append((total / len(choices))[:, 0].cpu())
    return torch.cat(outputs).numpy()
