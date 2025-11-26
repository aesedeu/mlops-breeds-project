import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim


class CustomTrainer(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        acc = (pred.argmax(dim=1) == y).float().mean()
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch):
        data, logits = batch
        preds = self(data)
        acc = (preds.argmax(dim=1) == logits).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        return {"test_acc": acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
