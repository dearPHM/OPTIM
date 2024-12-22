from math import ceil

import torch
from torch import nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True


#############################################
#            Network Components             #
#############################################


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels,
                         kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################


def make_net(widths, batchnorm_momentum, scaling_factor):
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(scaling_factor),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

############################################
#               Evaluation                 #
############################################


def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


#############################################
#       Whitening Conv Initialization       #
#############################################


def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()


def get_whitening_parameters(patches):
    n, c, h, w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(
        est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w, c, h, w).flip(0)


def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat(
        (eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                Lookahead                 #
############################################


class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)


############################################
#                Training                  #
############################################

def train(
        # run,
        model,
        train_loader, test_loader,
        batch_size, epochs, momentum, lr, wd, lr_biases,
        label_smoothing, whiten_bias_epochs,
        tta_level
):
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing, reduction='none')

    # if run == 'warmup':
    #     # The only purpose of the first run is to warmup, so we can use dummy data
    #     train_loader.labels = torch.randint(0, 10, size=(
    #         len(train_loader.labels),), device=train_loader.labels.device)

    total_train_steps = ceil(len(train_loader) * epochs)

    current_steps = 0

    norm_biases = [p for k, p in model.named_parameters()
                   if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters(
    ) if 'norm' not in k and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(
        param_configs, momentum=momentum, nesterov=True)

    def triangle(steps, start=0, end=0, peak=0.5):
        xp = torch.tensor([0, int(peak * steps), steps])
        fp = torch.tensor([start, 1, end])
        x = torch.arange(1+steps)
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])
        indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indices = torch.clamp(indices, 0, len(m) - 1)
        return m[indices] * x + b[indices]
    lr_schedule = triangle(total_train_steps, start=0.2, end=0.07, peak=0.23)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: lr_schedule[i])

    alpha_schedule = 0.95**5 * \
        (torch.arange(total_train_steps+1) / total_train_steps)**3
    lookahead_state = LookaheadState(model)

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    train_acc_collect, train_loss_collect = [], []
    val_acc_collect = []

    for epoch in range(ceil(epochs)):
        model[0].bias.requires_grad = (epoch < whiten_bias_epochs)

        ####################
        #     Training     #
        ####################

        starter.record()

        model.train()
        for inputs, labels in train_loader:

            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_steps += 1

            if current_steps % 5 == 0:
                lookahead_state.update(
                    model, decay=alpha_schedule[current_steps].item())

            if current_steps >= total_train_steps:
                if lookahead_state is not None:
                    lookahead_state.update(model, decay=1.0)
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) ==
                     labels).float().mean().item()
        train_loss = loss.item() / batch_size
        train_acc_collect.append(train_acc)
        train_loss_collect.append(train_loss)
        # val_acc = evaluate(model, test_loader, tta_level=0)
        val_acc = evaluate(model, test_loader, tta_level=tta_level)
        val_acc_collect.append(val_acc)
        # print_training_details(locals(), is_final_entry=False)
        # run = None  # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    # starter.record()
    # tta_val_acc = evaluate(
    #     model, test_loader, tta_level=tta_level)
    # ender.record()
    # torch.cuda.synchronize()
    # total_time_seconds += 1e-3 * starter.elapsed_time(ender)
    # epoch = 'eval'
    # print_training_details(locals(), is_final_entry=True)

    # print("total_time_seconds:", total_time_seconds)

    return train_acc_collect, train_loss_collect, val_acc_collect
