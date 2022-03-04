import torch
import numpy as np
from torch import nn, autograd
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = - torch.mean(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss  , 2e-8 * tv_loss
#         tv_loss = self.tv_loss(out_images)
        return image_loss , 0.001 * adversarial_loss , 0.006 * perception_loss

def mpute_gradient_penalty(use_cuda, netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
#     BATCH_SIZE = real_data.size(0)
    alpha = torch.rand(real_data.size()).uniform_(0,1)
#     alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 88, 88)
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_gradient_penalty(use_cuda, D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    if use_cuda:
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
class AdversarialLoss(nn.Module):
    def __init__(self, metric="mse"):
        super().__init__()
        self.register_buffer("label", torch.tensor(1.0))
        if metric == "mse":
            self.loss = nn.MSELoss()
        elif metric == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError()
        print("Adversarial loss:", self.loss.__class__.__name__)

    def __call__(self, input, target):
        loss = 0
        if isinstance(input, list):
            for i in input:
                target = (self.label * target).expand_as(i)
                loss += self.loss(i, target)
            loss /= len(input)
        else:
            target = (self.label * target).expand_as(input)
            loss += self.loss(input, target)
        return loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
