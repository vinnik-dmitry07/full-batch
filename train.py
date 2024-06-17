import math

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.plugins import MixedPrecisionPlugin
from pytorch_lightning_sam_callback import SAM
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

seed_everything(27)

BATCH_SIZE = 256
NUM_WORKERS = 0


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr, num_samples):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y, task='multiclass', num_classes=10)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    # noinspection PyUnresolvedReferences
    def configure_optimizers(self):
        # https://github.com/Liuhong99/Sophia/commit/a7e157229b71d58cf995d32854f1be15c265b350
        # from sophia import SophiaG
        # https://github.com/facebookresearch/madgrad/commit/bdbd2d760cb5e73f8f1acb287b3844a29f75282d
        # from madgrad import MADGRAD
        # https://github.com/lucidrains/lion-pytorch
        # from lion_pytorch import Lion
        # https://github.com/cybertronai/pytorch-lamb/commit/d3ab8dccf6717977c1ad0d6b95499f6b25bba41b
        # from pytorch_lamb import Lamb
        # https://github.com/facebookresearch/schedule_free/commit/c5fca72b76de6529a43d3959f2d8f0d0a1c8ce26
        # from schedulefree import SGDScheduleFree, AdamWScheduleFree
        from torch.optim import SGD, Adam, AdamW
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=5e-4,
        )
        steps_per_epoch = math.ceil(self.hparams.num_samples / BATCH_SIZE)
        scheduler_dict = {
            'scheduler': OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == '__main__':
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir='.',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    model = LitResnet(lr=0.05, num_samples=cifar10_dm.num_samples)

    # TODO https://github.com/Lightning-Universe/lightning-flash/blob/master/examples/image/learn2learn_img_classification_imagenette.py
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        # accumulate_grad_batches=4000 // 256,
        devices=[0],
        callbacks=[
            # https://github.com/ar90n/pytorch-lightning-sam-callback/commit/3068c2dede6e49c6461daf966e2da969d24257f8
            SAM(),
            # Comment `if trainer.lr_scheduler_configs` in pytorch_lightning.callbacks.stochastic_weight_avg.py:
            StochasticWeightAveraging(swa_epoch_start=0., swa_lrs=4e-7),
            LearningRateMonitor(logging_interval="step")
        ],
        # Does not work with used SAM callback, use default float32:
        # plugins=MixedPrecisionPlugin(precision=16, device='cuda'),
        # resume_from_checkpoint='lightning_logs/version_4/checkpoints/epoch=2665-step=7998.ckpt'
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, cifar10_dm)
