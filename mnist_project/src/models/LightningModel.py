from src.data.make_dataset import CorruptMnist
from torch import Tensor, nn, optim
from pytorch_lightning import LightningModule
import torch


class CNN_Model(LightningModule):
    """
    Basic neural network
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),  # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),  # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),  # [N, 8, 20]
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, 128), nn.Dropout(), nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        """Runs inference on input x
        Args:
            x: tensor with shape [N, 1, 28, 28]
        Returns:
            log_probs: tensor with log probabilities with shape [N, 10]
        """
        return self.classifier(self.backbone(x))

    def training_step(self, btach, btach_idx):
        data, target = btach
        preds = self(data)
        loss = self.criterion(preds, target)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    def train_dataloader(self):
        train_set = CorruptMnist(
            train=True, in_folder="../../data/raw", out_folder="../../data/processed")
        return torch.utils.data.DataLoader(train_set, batch_size=128)
