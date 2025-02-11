
import argparse
import matplotlib.pyplot as plt
import torch
import wandb
from src.data.make_dataset import CorruptMnist
from src.models.model import Model


wandb.init(project="mnist-project", entity="team-s20")


wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
}


def training() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=1e-3)
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    model = model.to(device)
    wandb.watch(model, log_freq=100)
    train_set = CorruptMnist(
        train=True, in_folder="../../data/raw", out_folder="../../data/processed")
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = 5
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x.to(device))
            loss = criterion(preds, y.to(device))
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
            wandb.log({"loss": loss})
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
    torch.save(model.state_dict(), "models/trained_model.pt")

    plt.plot(loss_tracker, "-")
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.savefig(f"reports/figures/training_curve.png")


if __name__ == "__main__":
    training()
