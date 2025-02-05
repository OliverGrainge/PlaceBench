import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50 
from .base import SingleStageMethod
import torchvision.transforms as T


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.avgpool = None
        self.model.fc = None
        self.model.layer4 = None
        self.out_channels = 1024

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPRAgg(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
    ):
        super().__init__()
        hw = in_h * in_w
        self.mix = nn.Sequential(
            *[
                FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
                for _ in range(mix_depth)
            ]
        )
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class MixVPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet()
        self.aggregator = MixVPRAgg(
            in_channels=1024,
            in_h=20,
            in_w=20,
            out_channels=1024,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=4,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x



class MixVPR(SingleStageMethod):
    def __init__(
        self,
        name="MixVPR",
        model=None,
        transform=T.Compose([
            T.Resize((320, 320), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ]),
        descriptor_dim=4096,
        search_dist="cosine",
    ):
        model = MixVPRModel()
        sd = torch.load(
            "/home/oliver/Documents/github/PlaceBench/methods/weights/MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(sd)

        super().__init__(name, model, transform, descriptor_dim, search_dist)
