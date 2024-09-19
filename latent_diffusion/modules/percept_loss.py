import torch as pt
from torch import nn
from torch.nn.functional import mse_loss
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    """
    Perceptual loss Module

    This module calculates Pececeptual loss using VGG19 for input and target images.
    Parameters:
        layers (`list[int]`): list of VGG layers to extract features from
        normalized (`bool`): if input is normalized to VGG19 stats
        scale (`float`): Scaling factor for Percept loss. Default = 0.1
        device (`torch.device`): compute device
    """
    def __init__(
            self,
            layers: list = [4, 9, 22],
            normalized: bool = True,
            scale: float = 0.1,
            device: pt.device = pt.device("cuda:0")
    ):
        super(PerceptualLoss, self).__init__()
        self.normalized = normalized
        self.scale = scale
        self.device = device
        self.vgg = vgg19(weights=VGG19_Weights)
        self.layers = nn.ModuleList()
        last_layer = 0
        for p in self.vgg.parameters():
            p.requires_grad = False
        for i in sorted(layers):
            self.layers.append(self.vgg.features[last_layer: i].eval())
            last_layer = i
        self.register_buffer(
            "mean", pt.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", pt.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
            self,
            input_image: pt.Tensor,
            target: pt.Tensor
    ):
        """
        Calculate Perceptual loss for given input (x) and target image
        All imahes should be normalized 0,1 or as per VGG19
        Args:
            input_image (`torch.Tensor`) :
                Input image of shape `[batch, channel, height, width]`
            target (`torch.Tensor`) :
                Target image of shape `[batch, channel, height, width]`
        Return:
            [`torch.Tensor`] : Perceptual loss
        """
        assert input_image.shape == target.shape
        assert input_image.shape[2] >= 32
        loss = pt.Tensor([0.]).to(self.device)

        if not self.normalized:
            # normalize image
            input_image = (input_image - self.mean) / self.std
            target = (target - self.mean) / self.std
        x = input_image
        y = target
        for _, layer in enumerate(self.layers):
            inp = pt.concat((x, y), dim=0)
            inp = layer(inp)
            x, y = inp.chunk(2)
            loss += mse_loss(x, y)
        return self.scale*loss
