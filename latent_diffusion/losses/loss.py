import torch.nn.functional as F

def gan_loss_hinge_dis(dis_fake, dis_real):
    loss_real = F.relu(1.0 - dis_real).mean()
    loss_fake = F.relu(1.0 + dis_fake).mean()
    return 0.5 * (loss_real + loss_fake)

def gan_loss_hinge_gen(dis_fake):
    return -dis_fake.mean()


