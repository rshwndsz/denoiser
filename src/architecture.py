import os
import wget
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
import pytorch_lightning as pl

import constants as C


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))

        # Initialize layers
        # nn.init.constant_(self.bn.weight.data, 1.25 * np.sqrt(in_channels))
        # nn.init.kaiming_normal_(self.conv.weight.data, nonlinearity='relu')


class ResidualBlock(nn.Module):
    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)
        
    def forward(self, x):
        residual = x
        out      = self.relu_conv1(x)
        out      = self.relu_conv2(out)
        out      = out + residual
        return out


class MemoryBlock(nn.Module):
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, long_term_mem):
        short_term_mem = []
        residual       = x
        for layer in self.recursive_unit:
            x = layer(x)
            short_term_mem.append(x)
        
        gate_out = self.gate_unit(torch.cat(long_term_mem + short_term_mem, 1))
        long_term_mem.append(gate_out)
        return gate_out


class MemNet(pl.LightningModule):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor     = BNReLUConv(channels, in_channels)
        self.dense_memory      = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i + 1) for i in range(num_memblock)]
        )

    def forward(self, x):
        residual      = x
        out           = self.feature_extractor(x)
        long_term_mem = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, long_term_mem)
        out = self.reconstructor(out)
        out = out + residual
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=C.INIT_LR)
        
    def loss_function(self, recon, clean):
        return F.mse_loss(recon, clean)

    def prepare_data(self):
        """Data prep. To be run once."""
        # Download & Extract training & validation dataset
        if os.path.isdir("BSD300"):
            logger.debug("BDS300 already exists.")
        else:
            # Download
            fname = wget.download(C.BSDS_URL)
            # Extract
            with tarfile.open(fname) as tar:
                tar.extractall()
            # Cleanup
            os.remove("BSDS300-images.tgz")

        # Download & Extract testing dataset
        if os.path.isdir("cifar-10"):
            logger.debug("cifar-10 already exists.")
            self.test_dataset = CIFAR10(root_dir="cifar-10", train=False,
                                        image_size=C.IMG_SIZE, sigma=C.SIGMA, download=False)
        else:
            # Download
            self.test_dataset = CIFAR10(root_dir="cifar-10", train=False,
                                        image_size=C.IMG_SIZE, sigma=C.SIGMA, download=True)
        
    def setup(self, stage):
        # Loading training + validation dataset
        trainval_dataset   = NoisyBSDS(root_dir=BSDS_ROOT, mode="train", image_size=C.IMG_SIZE, sigma=C.SIGMA)
        # Randomly split into 70-30 (Train-Val)
        self.train_dataset, self.val_dataset = td.random_split(trainval_dataset, 
                                                               [int(0.7*len(trainval_dataset)), 
                                                                int(0.3*len(trainval_dataset))])
        
    def train_dataloader(self):
        return td.DataLoader(
            self.train_dataset, batch_size=C.BATCH_SIZE,
            num_workers=C.NUM_WORKERS, pin_memory=C.USE_GPU,
            shuffle=True,
        )

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        recon        = self(noisy)
        loss         = self.loss_function(recon, clean)
        log          = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def val_dataloader(self):
        return td.DataLoader(
            self.val_dataset, batch_size=C.BATCH_SIZE,
            num_workers=C.NUM_WORKERS, pin_memory=C.USE_GPU,
            shuffle=False,
        )

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        recon        = self(noisy)
        val_loss     = self.loss_function(recon, clean)
        return {'val_loss': val_loss, 'recon': recon, 'noisy': noisy}

    def validation_epoch_end(self, outputs):
        # Compute average validation loss for current epoch
        avg_val_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        # Read the last reconstructed output and corresponding noisy input batch
        recon        = outputs[-1]['recon']
        noisy        = outputs[-1]['noisy']

        # Create a grid image for the batch
        n     = min(noisy.size(0), 8)
        cat   = torch.cat([noisy.view(noisy.size(0), C.IN_CHANNELS, *C.IMG_SIZE)[:n], 
                           recon.view(recon.size(0), C.IN_CHANNELS, *C.IMG_SIZE)[:n]])
        grid  = tv.utils.make_grid(cat.cpu(), nrow=n, padding=2, pad_value=0, 
                                   normalize=False, scale_each=False, range=None)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        # Log the grid image to Tensorboard
        self.logger.experiment.add_image(f'val_results_{self.current_epoch}', ndarr, dataformats='CHW')
        # Save a copy locally
        tv.utils.save_image(cat, nrow=n, fp=f"./val_results_{self.current_epoch}.jpg")

        # Log the average validation loss to tensorboard
        log = {'avg_val_loss': avg_val_loss}
        # Return `avg_val_loss` as `val_loss` for EarlyCheckpoint callback
        return { 'log': log , 'val_loss': avg_val_loss }

    def test_dataloader(self):
        return td.DataLoader(
            self.test_dataset, batch_size=int(C.IMG_SIZE[0] / 32) * 4, 
            num_workers=C.NUM_WORKERS, pin_memory=C.USE_GPU,
            shuffle=False,
        )

    def test_step(self, batch, batch_idx):
        n     = int(C.IMG_SIZE[0] / 32)
        clean = tv.utils.make_grid(batch, nrow=n, padding=0, normalize=False, 
                                   range=None, scale_each=False, pad_value=0)
        clean = clean.unsqueeze(dim=0)
        noisy = (clean + torch.tensor(2, dtype=clean.dtype) / 
                 torch.tensor(255, dtype=clean.dtype) * torch.tensor(C.SIGMA, dtype=clean.dtype) * torch.randn_like(clean))
        recon = self(noisy)

        return {'recon': recon, 'noisy': noisy, 'clean': clean, 'id': f'{self.current_epoch}--{batch_idx}'}

    def test_step_end(self, outputs):
        n     = 3
        recon = outputs['recon']
        noisy = outputs['noisy']
        clean = outputs['clean']

        cat   = torch.cat([noisy.view(noisy.size(0), C.IN_CHANNELS, *C.IMG_SIZE)[:n], 
                           recon.view(recon.size(0), C.IN_CHANNELS, *C.IMG_SIZE)[:n],
                           clean.view(clean.size(0), C.IN_CHANNELS, *C.IMG_SIZE)[:n]])
        grid  = tv.utils.make_grid(cat.cpu(), nrow=3, padding=4, pad_value=0, 
                                   normalize=False, scale_each=False, range=None)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        self.logger.experiment.add_image(f"test_results_{outputs['id']}", ndarr, dataformats='CHW')
                
        tv.utils.save_image(cat, nrow=3, padding=4, pad_value=0,
                            normalize=False, scale_each=False, range=None,
                            fp=f"./test_result_{outputs['id']}.jpg")

    def test_epoch_end(self, outputs):
        pass

