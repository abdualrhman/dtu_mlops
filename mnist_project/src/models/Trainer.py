import os
from catboost import train
from pytorch_lightning import Trainer
from src.data.make_dataset import CorruptMnist
from src.models.LightningModel import CNN_Model
import torch


train_set = CorruptMnist(
    train=True, in_folder="../../data/raw", out_folder="../../data/processed")
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
test_set = CorruptMnist(
    train=False, in_folder="../../data/raw", out_folder="../../data/processed")
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)

model = CNN_Model()
trainer = Trainer(default_root_dir=os.getcwd())
trainer.fit(model, train_dataloader)
train.test(model, test_dataloader)
