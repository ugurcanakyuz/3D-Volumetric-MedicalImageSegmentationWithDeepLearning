import torchio as tio
import tqdm


class Trainer2D:
    """This class consist of fit method for training. fit method train the model with given data.

    Example:
        trainer = Trainer2D(model, train_loader, optimizer, criterion, num_epochs, scheduler)
        avg_train_loss = trainer.fit()
    """
    def __init__(self, model, train_loader, optimizer, criterion, total_epochs, scheduler=None):
        self.curr_epoch = 0

        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_epochs = total_epochs
        self.train_loader = train_loader

        model.train()

    def fit(self):
        """Trains the model with given data.

        Returns
        -------
        avg_train_loss: float
        """

        avg_train_loss = None
        device = next(self.model.parameters()).device
        count_forward = 0
        running_loss = 0.0
        bs_2d = 16

        prog_bar = tqdm.tqdm(enumerate(self.train_loader),
                             total=int(len(self.train_loader) / self.train_loader.batch_size))
        prog_bar.set_description(f"Epoch [{self.curr_epoch + 1}/{self.total_epochs}]")
        prog_bar.set_postfix_str(f'Loss: {avg_train_loss}')

        for i, (image, mask) in prog_bar:
            image = image.to(device)  # [bs,x,y,z]
            mask = mask.to(device)  # [x,y,z]

            # Slice 3D image. It's like splitting 3D images into batches.
            for slice_ix in range(0, image.shape[1], bs_2d):
                start = slice_ix
                stop = slice_ix + bs_2d

                if stop > image.shape[1]:
                    stop = image.shape[1]

                slice_image = image[:, start:stop]
                slice_mask = mask[:, start:stop]
                slice_image = slice_image.view(-1, 1, 256, 256)
                slice_mask = slice_mask.view(-1, 1, 256, 256)

                outputs = self.model(slice_image.float())

                loss = self.criterion(outputs, slice_mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Sum losses and dice scores for all predictions.
                running_loss += loss.item()
                count_forward += 1

            avg_train_loss = running_loss / count_forward
            prog_bar.set_postfix_str(f'Loss: {avg_train_loss:.4f}')

        self.curr_epoch += 1

        if self.scheduler:
            # Update scheduler.
            self.scheduler.step()

        return avg_train_loss


class Trainer3D:
    def __init__(self, model, train_loader, optimizer, criterion, patch_size, total_epochs, scheduler=None):
        self.curr_epoch = 0

        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.patch_size = patch_size
        self.scheduler = scheduler
        self.total_epochs = total_epochs
        self.train_loader = train_loader

        model.train()

    def fit(self):
        avg_loss = None
        device = next(self.model.parameters()).device
        epoch_loss = []
        running_loss = []  # This is only for loss indicator.

        # sampler = tio.data.WeightedSampler(patch_size_, "sampling_map")
        # sampler = tio.data.GridSampler(patch_size=patch_size_)
        sampler = tio.data.UniformSampler(patch_size=self.patch_size)

        prog_bar = tqdm.tqdm(enumerate(self.train_loader),
                             total=int(len(self.train_loader) / self.train_loader.batch_size))
        prog_bar.set_description(f"Epoch [{self.curr_epoch + 1}/{self.total_epochs}]")
        prog_bar.set_postfix_str(f'Loss: {avg_loss}')

        for i, (image, mask) in prog_bar:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask),
                # sampling_map=tio.Image(tensor=mask, type=tio.SAMPLING_MAP),       # Mask is more stable for sampling.
            )

            for j, patch in enumerate(sampler(subject)):
                patch_mask = patch["mask"].data.unsqueeze(1).to(device)  # [bs,1,x,y,z]
                patch_image = patch["image"].data
                patch_image = patch_image.unsqueeze(1).to(device)  # [bs,1,x,y,z]

                outputs = self.model(patch_image.float())
                loss = self.criterion(outputs, patch_mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Sum losses scores for all predictions.
                running_loss.append(loss.item())

                if j > 7:
                    break

            avg_loss = sum(running_loss) / len(running_loss)
            running_loss = []
            epoch_loss.append(avg_loss)
            prog_bar.set_postfix_str(f'Loss: {sum(epoch_loss) / len(epoch_loss):.4f}')

        self.curr_epoch += 1

        if self.scheduler:
            # Update scheduler.
            self.scheduler.step()

        return sum(epoch_loss) / len(epoch_loss)
