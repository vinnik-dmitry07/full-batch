import math

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import MixedPrecisionPlugin
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
        from sophia import SophiaG
        from madgrad import MADGRAD
        from lion_pytorch import Lion
        from pytorch_lamb import Lamb
        from torch.optim import SGD, Adam, AdamW

        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=5e-3,
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

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        # accumulate_grad_batches=4000 // 256,
        devices=[0],
        callbacks=LearningRateMonitor(logging_interval="step"),
        plugins=MixedPrecisionPlugin(precision=16, device='cuda'),
        # resume_from_checkpoint='lightning_logs/version_4/checkpoints/epoch=2665-step=7998.ckpt'
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)
