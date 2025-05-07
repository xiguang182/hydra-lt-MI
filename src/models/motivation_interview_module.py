from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import MulticlassF1Score as F1


class MILitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        class_weights: float = 7.0,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        # add class_weights to the loss function

        self.class_weights = torch.tensor([1.0, class_weights])
        smoothed_weights = self.class_weights / self.class_weights.sum()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        # self.criterion = torch.nn.CrossEntropyLoss(weight=smoothed_weights)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        self.train_f1 = F1(num_classes=2, multidim_average='global', average=None)
        self.val_f1 = F1(num_classes=2, multidim_average='global', average=None)
        self.test_f1 = F1(num_classes=2, multidim_average='global', average=None)

        # metric objects for calculating and averaging precision and f1 score across batches
        # self.precision = Precision(num_classes=2)
        # self.f1score = F1(num_classes=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_sustain_best = MaxMetric()
        self.val_f1_change_best = MaxMetric()
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x1, x2, x3)
    
    def on_fit_start(self):
        # self.net.to(self.device)
        print(f"Net is on device: {next(self.net.parameters()).device}")
        print(self.net)

    def on_train_start(self) -> None:
        # print("on_train_start called")
        self.net.to(self.device)
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        self.val_f1.reset()
        self.val_f1_sustain_best.reset()
        self.val_f1_change_best.reset()

# best practice for multiple input dataloader?
# might need to change the input batch to a tuple of arbitrary length and unpack it with considitions. For different input numbers like only one person, only language, etc.
# Currently it takes 4 tensors, 3 inputs and 1 target.Refer playground snippet for arbitrary number of inputs.
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of RoBERTa and two persons' Openface and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # if dataset is (x1, x2, x3, y), will it be Batch_size x (x1, x2, x3, y) ? or (Batch_size x x1, Batch_size x x2, Batch_size x x3, Batch_size x y) ?
        """
        In PyTorch, when you use a DataLoader with a dataset structured as (x1, x2, x3, y), the DataLoader will return batches in the format (Batch_size x x1, Batch_size x x2, Batch_size x x3, Batch_size x y).
        For cross entropy loss in PyTorch, the target tensor can simply have the shape (Batch_size), while the input is expected to have the shape (Batch_size, num_classes).
        """
        x1, x2, x3, y = batch
        logits = self.forward(x1, x2, x3)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # print(logits,preds,y)
        # print(f"Loss function weights: {self.criterion.weight}")
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss.update(loss)
        self.train_acc.update(preds, targets)
        self.train_f1.update(preds, targets)
        # print(self.train_f1.compute(),self.train_f1[0])
        # tensor([0.0833, 0.6944], device='cuda:0') CompositionalMetric<lambda>(MulticlassF1Score(),None))
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        f1_per_class = self.train_f1.compute()
        acc = self.train_acc.compute()  # get current train acc
        
        # log each class's F1 explicitly
        self.log("train/f1_sustain_talk", f1_per_class[0], prog_bar=True)
        self.log("train/f1_change_talk", f1_per_class[1], prog_bar=True)

        self.log("train/loss", self.train_loss.compute(), prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)

        # reset running metrics
        self.train_loss.reset()
        self.train_acc.reset()
        self.train_f1.reset()


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        
        # print('preds',preds)
        # print('label',targets)

        # update metrics
        self.val_loss.update(loss)
        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)

        # log running metrics (Lightning will call .compute() at epoch end)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # compute once
        f1_per_class = self.val_f1.compute()
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        
        # log each class's F1 explicitly
        self.log("val/f1_sustain_talk", f1_per_class[0], prog_bar=True)
        self.log("val/f1_change_talk", f1_per_class[1], prog_bar=True)

        # update best metrics
        
        self.val_f1_sustain_best(f1_per_class[0])
        self.val_f1_change_best(f1_per_class[1])
        self.log("val/f1_change_talk_best", self.val_f1_change_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1_sustain_talk_best", self.val_f1_sustain_best.compute(), sync_dist=True, prog_bar=True)

        self.log("val/loss", self.val_loss.compute(), prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

        # reset running metrics
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        

        
        

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        
        # update and log metrics

        self.test_loss.update(loss)
        self.test_acc.update(preds, targets)
        self.test_f1.update(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        f1_per_class = self.test_f1.compute()
        acc = self.test_acc.compute()  # get current train acc
        
        # log each class's F1 explicitly
        self.log("test/f1_sustain_talk", f1_per_class[0], prog_bar=True)
        self.log("test/f1_change_talk", f1_per_class[1], prog_bar=True)

        self.log("test/loss", self.test_loss.compute(), prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MILitModule(None, None, None, None)
