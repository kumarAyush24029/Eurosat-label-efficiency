"""
models.py  -  the neural networks we compare.

We deliberately choose three model "families" because each represents a different
idea, and the project question is how they behave when labels are scarce:

  1. SmallCNN  - a tiny convolutional network trained from scratch.
                 Our BASELINE: shows what you get with no outside knowledge.

  2. ResNet-50 - a deep convolutional network with "skip connections" (the standard
                 workhorse of image classification). We run it two ways:
                   * scratch  : random weights   -> needs lots of labels
                   * imagenet : weights pre-trained on 1.2M ImageNet photos
                                (TRANSFER LEARNING) -> already knows edges, textures,
                                shapes, so it needs far fewer labels.
                 Comparing the two quantifies the value of pre-training.

  3. ViT-S/16  - a Vision Transformer (small). Instead of convolutions it cuts the
                 image into patches and uses attention. Transformers are
                 data-hungry from scratch but very strong when pre-trained, so they
                 are an interesting contrast to CNNs in the low-label regime.
                 We use the ImageNet-21k-pretrained version.

Using extra spectral bands: `in_chans` changes the number of input channels. For
pretrained models, timm automatically adapts the first layer (it reuses the RGB
filters for the new channels) so we can still benefit from pre-training with 13 bands.
"""
import torch.nn as nn

# Short name used on the command line -> exact model name in the `timm` library.
TIMM_NAMES = {
    "resnet18": "resnet18",                                   # small ResNet (handy for quick tests)
    "resnet50": "resnet50",
    "vit_s": "vit_small_patch16_224.augreg_in21k_ft_in1k",    # ViT-Small, patch size 16
}


class SmallCNN(nn.Module):
    """A simple 4-stage CNN (~1.2M parameters), trained from scratch.

    Each stage = two [3x3 conv -> BatchNorm -> ReLU] layers followed by 2x2 max-pool,
    which halves the image size (64 -> 32 -> 16 -> 8 -> 4) while the number of
    feature channels doubles (32 -> 64 -> 128 -> 256).
    At the end we average each channel over the whole image ("global average pooling")
    and use one linear layer to produce the 10 class scores.
    """

    def __init__(self, in_chans=3, n_classes=10, w=32):
        super().__init__()
        chans = [in_chans, w, 2 * w, 4 * w, 8 * w]          # channels at each stage
        layers = []
        for i in range(4):
            layers += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(chans[i + 1]),                # keeps activations well-scaled
                nn.ReLU(True),
                nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(chans[i + 1]),
                nn.ReLU(True),
                nn.MaxPool2d(2),                             # downsample by 2
            ]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(chans[-1], n_classes)          # final classifier

    def forward(self, x):
        # features -> (B, 256, 4, 4); mean over H,W -> (B, 256); head -> (B, 10) class scores
        return self.head(self.features(x).mean((2, 3)))


def build_model(name, pretrained, in_chans=3, n_classes=10, img_size=64):
    """Create a model by short name.

    name       : "cnn", "resnet18", "resnet50" or "vit_s"
    pretrained : True -> load ImageNet weights (transfer learning); False -> random init
    in_chans   : number of input channels (3 for RGB, up to 13 for multispectral)
    n_classes  : EuroSAT has 10 land-use classes
    img_size   : input resolution (only ViT needs to know; its position embeddings
                 are resized automatically to fit)
    """
    if name == "cnn":
        return SmallCNN(in_chans, n_classes)
    import timm                                              # imported here so the CNN needs no timm
    kw = dict(pretrained=pretrained, num_classes=n_classes, in_chans=in_chans)
    if name.startswith("vit"):
        kw["img_size"] = img_size
    return timm.create_model(TIMM_NAMES[name], **kw)
