import torch
from torch.utils.data import DataLoader
import tqdm


class Trainer3D:
    """Trainer for 3D Data and models.

    Example:
        trainer = Trainer2D(criterion, model, optimizer, total_epochs, train_loader)
        avg_loss = trainer.fit()
    """

    def __init__(self, criterion, model, optimizer, total_epochs, train_loader, scheduler=None):
        """Creates 3D Trainer object.

        Parameters
        ----------
        criterion: modules.LossFunction
            To calculate loss value of 2D Data.
        model: modules.UNet
            3D UNet model implemented in UNet.
        optimizer: torch.optim.SGD or others
            Optimizer object of the torch.optim module.
        total_epochs: int
            Total epoch counts of the training.
        train_loader: DataLoader
            3D Data loader.
        scheduler: torch.optim.lr_scheduler.StepLR
            Learning rate scheduler object of the torch.optim.lr_scheduler module.

        Returns
        -------
        None
        """

        self.curr_epoch = 0

        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_epochs = total_epochs
        self.train_loader = train_loader

    def fit(self):
        """Fit the model to the 3D Data.

        Returns
        -------
        float
            Average loss value of the epoch.
        """

        avg_loss = None
        device = next(self.model.parameters()).device
        epoch_loss = []
        running_loss = []  # This is only for loss indicator.

        prog_bar = tqdm.tqdm(enumerate(self.train_loader),
                             total=int((len(self.train_loader)) / self.train_loader.batch_size))
        prog_bar.set_description(f"Epoch [{self.curr_epoch + 1}/{self.total_epochs}]")
        prog_bar.set_postfix_str(f'Loss: {avg_loss}')

        self.model.train()
        for i, subject in prog_bar:
            patch_mri = subject['mri']['data'].to(device)  # [bs,1,x,y,z]
            patch_mask = subject['mask']['data'].to(device)  # [bs,1,x,y,z]

            outputs = self.model(patch_mri.float())
            loss = self.criterion(outputs, patch_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())

            if (i+1) % 8 == 0:  # 8 is number of patches per volume
                # Sum losses scores for all predictions.
                avg_loss = sum(running_loss) / len(running_loss)
                running_loss = []
                epoch_loss.append(avg_loss)
                prog_bar.set_postfix_str(f'Loss: {sum(epoch_loss) / len(epoch_loss):.4f}')

        self.curr_epoch += 1

        if self.scheduler:
            # Update scheduler.
            self.scheduler.step()

        return sum(epoch_loss) / len(epoch_loss)
