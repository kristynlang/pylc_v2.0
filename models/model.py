"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Base Model Class
File: model.py
"""

import os
import sys
import torch
import numpy as np
from models.architectures.unet import UNet
from models.architectures.res_unet import ResUNet
from models.architectures.deeplab import DeepLab
from models.modules.loss import MultiLoss, RunningLoss
from models.modules.evaluator import Evaluator
from models.modules.checkpoint import Checkpoint
from numpy import random
from config import cf


class Model:
    """
    Abstract model for Pytorch network configuration.
    Uses Pytorch Model class as superclass

      Parameters
      ------
      cf: dict
         User configuration settings.
    """

    def __init__(self):

        super(Model, self).__init__()

        # input configuration
        self.id = cf.id
        self.n_classes = cf.n_classes
        self.ch = cf.ch
        self.arch = cf.arch

        # activation functions
        self.activ_func = activation

        # build network
        self.net = None
        self.crop_target = False

        # If loading, load model from '.pth' file, otherwise build new
        if cf.load:
            self.load()
        else:
            self.build()

        # initialize global iteration counter
        self.iter = 0

        # initialize preprocessed metadata -> parameters
        self.metadata = self.init_metadata()
        self.cls_weight = self.metadata.item().get('weights')
        self.px_mean = self.metadata.item().get('px_mean')
        self.px_std = self.metadata.item().get('px_std')

        # load loss handlers
        self.crit = MultiLoss(self.cls_weight)
        self.loss = RunningLoss()

        # initialize run parameters
        self.epoch = 0
        self.optim = self.init_optim()
        self.sched = self.init_sched()

        # initialize training checkpoint and test evaluator
        self.checkpoint = Checkpoint()
        self.evaluator = Evaluator()

    def load(self):
        """
        Loads pretrained model for evaluation.
        """
        model_data = self.evaluator.load(cf.load)
        try:
            self.metadata = model_data["metadata"]
            self.net.load_state_dict(model_data["model"])
        except:
            print('An error occurred loading pretrained model at: \n{}'.format(
                cf.load))
            exit()

    def build(self):
        """
        Builds neural network model from configuration settings.
        """

        # UNet
        if self.arch == 'unet':
            self.net = UNet(
                in_channels=self.ch,
                n_classes=self.n_classes,
                up_mode=cf.up_mode,
                activ_func=self.activ_func('selu'),
                normalizer=Normalizer('instance'),
                dropout=cf.dropout
            )
            self.net = self.net.to(cf.device)
            self.crop_target = True

        # Alternate Residual UNet
        elif self.arch == 'resunet':
            self.net = ResUNet(
                in_channels=self.ch,
                n_classes=self.n_classes,
                up_mode=cf.up_mode,
                activ_func=self.activ_func('relu'),
                # normalizer=Normalizer('layer'),
                dropout=cf.dropout
            )
            self.net = self.net.to(cf.device)
            self.crop_target = True

        # DeeplabV3+
        elif self.arch == 'deeplab':
            self.net = DeepLab(
                activ_func=self.activ_func('relu'),
                normalizer=torch.nn.BatchNorm2d,
                backbone=cf.backbone,
                n_classes=self.n_classes,
                in_channels=self.ch,
                pretrained=cf.pretrained
            )
            self.net = self.net.to(cf.device)

        # Unknown model requested
        else:
            print('Model {} not available.'.format(self.cf.model))
            exit(1)

        # Enable CUDA
        if torch.cuda.is_available():
            print("\nCUDA enabled.")

        # Parallelize model on multiple GPUs (disabled)
        if torch.cuda.device_count() > 1:
            print("\t{} GPUs in use.".format(torch.cuda.device_count()))
            # self.net = torch.nn.DataParallel(self.net)

        # Check multiprocessing enabled
        if torch.utils.data.get_worker_info():
            print('\tMulti-process data loading: {} workers enabled.'.format(
                torch.utils.data.get_worker_info().num_workers))

    def resume(self):
        """
        Check for existing checkpoint. If exists, resume from
        previous training. If not, delete the checkpoint.
        """
        if self.cf.resume:
            checkpoint_data = self.checkpoint.load()
            self.epoch = checkpoint_data['epoch']
            self.iter = checkpoint_data['iter']
            self.metadata = checkpoint_data["metadata"]
            self.net.load_state_dict(checkpoint_data["model"])
            self.optim.load_state_dict(checkpoint_data["optim"])
        else:
            self.checkpoint.reset()

    def init_metadata(self):
        """
         Retrieve preprocessed metadata for training database.
        """

        path = os.path.join(self.cf.md_dir, self.cf.db + '.npy')

        # select dataset metadata file
        if os.path.isfile(path):
            print("\nLoading dataset metadata from {}.".format(path))
            return np.load(path, allow_pickle=True)
        else:
            print("Error: Metadata file {} not found. Parameters could not be initialized.".format(path))
            sys.exit(0)

    def init_optim(self):

        # select optimizer
        if self.cf.optim == 'adam':
            return torch.optim.AdamW(self.net.parameters(), lr=self.cf.lr, weight_decay=cf.weight_decay)
        elif self.cf.optim == 'sgd':
            return torch.optim.SGD(self.net.parameters(), lr=self.cf.lr, momentum=cf.momentum)
        else:
            print('Optimizer is not defined.')
            exit()

    def init_sched(self):

        # (Optional) Scheduled learning rate step
        if self.cf.sched == 'step_lr':
            return torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=cf.gamma)
        elif self.cf.sched == 'cyclic_lr':
            return torch.optim.lr_scheduler.CyclicLR(self.optim, cf.lr_min, cf.lr_max, step_size_up=2000)
        elif self.cf.sched == 'anneal':
            steps_per_epoch = int(self.cf.clip * 29000 // self.cf.batch_size)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=cf.lr_max,
                                                            steps_per_epoch=steps_per_epoch,
                                                            epochs=self.cf.n_epochs)
        else:
            print('Optimizer scheduler is not defined.')
            exit()

    def train(self, x, y):

        """model training step"""

        # apply random vertical flip
        if bool(random.randint(0, 1)):
            x = torch.flip(x, [3])
            y = torch.flip(y, [2])

        # normalize input [NCWH]
        x = self.normalize_image(x)
        x = x.to(cf.device)
        y = y.to(cf.device)

        # crop target mask to fit output size (e.g. UNet model)
        if self.cf.model == 'unet':
            y = y[:, cf.crop_left:cf.crop_right, cf.crop_up:cf.crop_down]

        # stack single-channel input tensors (deeplab)
        if self.ch == 1 and self.cf.model == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # forward pass
        y_hat = self.net.forward(x)

        # compute losses
        loss = self.crit.forward(y_hat, y)

        self.loss.intv += [(self.crit.ce.item(), self.crit.dsc.item(), self.crit.fl.item())]

        # zero gradients, compute, step, log losses,
        self.optim.zero_grad()
        loss.backward()

        # in-place normalization of gradients
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optim.step()

        if self.iter % self.cf.report == 0:
            self.log()

        # log learning rate
        self.loss.lr += [(self.iter, self.get_lr())]

        self.iter += 1

    def eval(self, x, y, test=False):

        """model test/validation step"""

        self.net.eval()

        # normalize
        x = self.normalize_image(x)
        x = x.to(cf.device)
        y = y.to(cf.device)

        # crop target mask to fit output size (UNet)
        if self.cf.model == 'unet':
            y = y[:, cf.crop_left:cf.crop_right, cf.crop_up:cf.crop_down]

        # stack single-channel input tensors (Deeplab)
        if self.ch == 1 and self.cf.model == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            ce = self.crit.ce_loss(y_hat, y).cpu().numpy()
            dice = self.crit.dice_loss(y_hat, y).cpu().numpy()
            focal = self.crit.focal_loss(y_hat, y).cpu().numpy()
            self.loss.intv += [(ce, dice, focal)]

        if test:
            self.evaluator.results += [y_hat]

    def test(self, x):

        """model test forward"""

        # normalize
        x = self.normalize_image(x, default=self.cf.normalize_default)
        x = x.to(cf.device)

        # stack single-channel input tensors (Deeplab)
        if self.ch == 1 and self.cf.model == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            self.evaluator.results += [y_hat]

    def log(self):

        """log ce/dice losses at defined intervals"""

        self.loss.log(self.iter, self.net.training)
        self.loss.save()

    def save(self, test=False):

        """save output tiles to file"""

        if test:
            self.evaluator.save()
            self.loss.save()
        else:
            self.checkpoint.save(self, is_best=self.loss.is_best)

    def save_image(self):

        """save predicted mask image to file"""

        self.evaluator.save_image()

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def normalize_image(self, img, default=False):
        """ Normalize input image data [NCWH]
            - uses precomputed mean/std of pixel intensities
        """
        if default:
            return torch.tensor((img.numpy().astype('float32') - cf.px_mean_default) / cf.px_std_default)
        if img.shape[1] == 1:
            mean = np.mean(self.px_mean.numpy())
            std = np.mean(self.px_std.numpy())
            return torch.tensor((img.numpy().astype('float32') - mean) / std) / 255
        else:
            return ((img - self.px_mean[None, :, None, None]) / self.px_std[None, :, None, None]) / 255

    def print_settings(self):
        """
        Prints model configuration settings to screen.
        """

        print("\n------\nModel Settings")
        print("\tID {}".format(self.id))
        print("\tDatabase: {}".format(self.cf.db))
        print("\tModel: {}".format(self.cf.arch))
        # show encoder backbone for Deeplab
        if self.cf.arch == 'deeplab':
            print("\tBackbone: {}".format(self.cf.backbone))
        print('\tInput channels: {}'.format(self.cf.ch))
        print('\tOutput channels (classes): {}'.format(self.n_classes))
        print("------\n")

    def summary(self):
        """
        Prints model parameters to screen.
        """
        try:
            from torchsummary import summary
            summary(self.net, input_size=(self.ch, cf.input_size, cf.input_size))
        except ImportError as e:
            print('Summary not available.')
            pass  # module doesn't exist


class Normalizer:
    """
    Class defines model normalizing functions.

      Parameters
      ------
      norm_type: str
         Type of normalizer to use.
    """

    def __init__(self, norm_type):
        self.type = norm_type

    def apply(self, n_features):
        """
        Network layer normalization functions
        """
        return torch.nn.ModuleDict([
            ['batch', torch.nn.BatchNorm2d(n_features)],
            ['instance', torch.nn.InstanceNorm2d(n_features)],
            ['layer', torch.nn.LayerNorm(n_features)],
            ['syncbatch', torch.nn.SyncBatchNorm(n_features)],
            ['none', None]
        ])[self.type]


def activation(active_type):
    """
    Network layer activation functions

    Paramters
    ------
    active_type: str
        Activation function key.
    """
    return {
        'relu': torch.nn.ReLU(inplace=True),
        'leaky_relu': torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'selu': torch.nn.SELU(inplace=True),
        'none': torch.nn.Identity()
    }[active_type]