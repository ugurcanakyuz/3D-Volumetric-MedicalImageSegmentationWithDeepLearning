import torch
import torch.nn.functional as F
import torchio as tio
import tqdm

from torch.utils.data import DataLoader
from modules.Utils import calculate_dice_score, create_onehot_mask


class Evaluator2D:
    """Evaluator of 2D Data. Loaded 3D MRI Data slices into 2D stacks of batches and
    calculates loss and DSC of batches per class.

    Example:
        evaluator = Evaluator2D(model, model, val_loader)
        avg_score = evaluator.evaluate()
    """

    def __init__(self, criterion, model, val_loader):
        """Creates 2D evaluator object.

        Parameters
        ----------
        criterion: modules.LossFunctions
            To calculate loss value of 2D Data.
        model: modules.UNet
            2D UNet model implemented in UNet.
        val_loader: DataLoader
            3D Data loader.

        Returns
        -------
        None
        """

        self.model = model
        self.val_loader = val_loader
        self.val_criterion = criterion

        self.bs_2d = 16
        self.device = next(model.parameters()).device

        out_channels = model.out.out_channels
        sample = next(iter(val_loader))
        shape = tuple(sample['mri']['data'].shape[1:])
        self.output_shape = (val_loader.batch_size, out_channels, *shape)

    def evaluate(self):
        """Calculates loss and dice score for each class.

        Parameters
        ----------
        None.

        Returns
        -------
        avg_loss: float
            Average loss of the validation set.
        dice_scores: Tensor
            Dice scores of each subject.
        """

        avg_val_loss = None
        epoch_loss = []
        running_losses = []
        dice_scores = []

        prog_bar = tqdm.tqdm(enumerate(self.val_loader),
                             total=int(len(self.val_loader) / self.val_loader.batch_size))
        prog_bar.set_description(f"Validation ")
        prog_bar.set_postfix_str(f'Loss: {avg_val_loss}')

        self.model.eval()
        with torch.no_grad():
            for i, subject in prog_bar:
                mri = subject['mri']['data'].to(self.device)          # [bs,1,x,y,z], bs=1 due to memory limit
                mask = subject['mask']['data'].to(self.device)        # [bs,1,x,y,z], bs=1 due to memory limit

                # Slice 3D image. It's like splitting 3D images into batches.
                for slice_ix in range(0, mri.shape[2], self.bs_2d):
                    start = slice_ix
                    stop = slice_ix + self.bs_2d

                    if stop > mri.shape[2]:
                        stop = mri.shape[2]

                    stack_mri = mri[:, :, start:stop].squeeze(0)        # start:stop, For example 0:16, 16:32, 32:64 .
                    stack_mask = mask[:, :, start:stop].squeeze(0)
                    stack_mri = stack_mri.permute(1, 0, 2, 3)           # [bs_2d, 1, y,z]
                    stack_mask = stack_mask.permute(1, 0, 2, 3)         # [bs_2d, 1, y,z]

                    output = self.model(stack_mri.float())

                    # Criterion accepts raw logits so softmax has not been applied to predicted mask yet.
                    val_loss = self.val_criterion(output, stack_mask)
                    running_losses.append(val_loss.item())

                    # To calculate Dice Score get softmax applied predicted mask.
                    pred_mask = F.softmax(output, dim=1)

                    # Create one hot encoded mask of original mask.
                    one_hot_mask = create_onehot_mask(pred_mask.shape, stack_mask)

                    # Convert class probabilities to actual class labels.
                    pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)

                    # Create one hot encoded mask of predicted mask.
                    pred_mask = create_onehot_mask(one_hot_mask.shape, pred_mask)

                    scores = calculate_dice_score(pred_mask, one_hot_mask)
                    dice_scores.append(scores)

                avg_loss = sum(running_losses) / len(running_losses)
                running_losses = []
                epoch_loss.append(avg_loss)
                prog_bar.set_postfix_str(f'Loss: {sum(epoch_loss) / len(epoch_loss):.4f}')

            avg_loss = sum(epoch_loss) / len(epoch_loss)

        return avg_loss, dice_scores


class Evaluator3D:
    """This class consist of evaluation method. Evaluate method calculates dice score per channel.

    Example:
        evaluator = Evaluator3D(model, model, val_loader)
        avg_score = evaluator.evaluate()
    """

    def __init__(self, criterion, model, patch_size, val_loader):
        """Creates 3D evaluator object.

        Parameters
        ----------
        criterion: modules.LossFunctions
            To calculate loss value of 3D Data.
        model: modules.UNet
            3D UNet model implemented in UNet.
        val_loader: DataLoader
            3D Data loader.

        Returns
        -------
        None
        """

        self.model = model
        self.patch_size = patch_size
        self.val_loader = val_loader
        self.device = next(model.parameters()).device
        self.val_criterion = criterion

        out_channels = model.out.out_channels
        sample = next(iter(val_loader))
        shape = tuple(sample['mri']['data'].shape[1:])
        self.output_shape = (val_loader.batch_size, out_channels, *shape)

    def evaluate(self):
        """Calculates dice score for each class.

        Parameters
        ----------
        None.

        Returns
        -------
        avg_loss: float
            Average loss of the validation set.
        dice_scores: Tensor
            Dice scores of each subject.
        """

        avg_val_loss = None
        epoch_loss = []
        overlap_mode_ = 'average'
        running_losses = []
        dice_scores = []

        prog_bar = tqdm.tqdm(enumerate(self.val_loader),
                             total=int(len(self.val_loader) / self.val_loader.batch_size))
        prog_bar.set_description(f"Validation ")
        prog_bar.set_postfix_str(f'Loss: {avg_val_loss}')

        self.model.eval()
        with torch.no_grad():
            for i, subject in prog_bar:
                mri = subject['mri']['data']
                mask = subject['mask']['data']
                subject = tio.Subject(image=tio.ScalarImage(tensor=mri.squeeze(0)),
                                      mask=tio.LabelMap(tensor=mask.squeeze(0)))
                sampler = tio.data.GridSampler(subject=subject, patch_size=self.patch_size)
                aggregator = tio.data.GridAggregator(sampler, overlap_mode=overlap_mode_)

                for j, patch in enumerate(sampler(subject)):
                    patch_mri = patch["image"].data.unsqueeze(1).to(self.device)  # [bs,1,x,y,z]

                    output = self.model(patch_mri.float())
                    aggregator.add_batch(output, patch["location"].unsqueeze(0))

                output = aggregator.get_output_tensor().unsqueeze(0)
                mask = mask

                # Validation loss calculated on the aggregated so whole predicted mask.
                # Criterion accepts raw logits so softmax has not been applied to predicted mask yet.
                val_loss = self.val_criterion(output, mask)
                running_losses.append(val_loss.item())

                # To calculate Dice Score get softmax applied predicted mask.
                # Actually, no need for softmax, argmax will be enough for calculations.
                pred_mask = F.softmax(output, dim=1)

                # Create one hot encoded mask of original mask.
                one_hot_mask = create_onehot_mask(pred_mask.shape, mask)

                # Convert class probabilities to actual class labels.
                pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)

                # Create one hot encoded mask of predicted mask.
                pred_mask = create_onehot_mask(one_hot_mask.shape, pred_mask)

                scores = calculate_dice_score(pred_mask, one_hot_mask)
                dice_scores.append(scores)

                avg_loss = sum(running_losses) / len(running_losses)
                running_losses = []
                epoch_loss.append(avg_loss)
                prog_bar.set_postfix_str(f'Loss: {sum(epoch_loss) / len(epoch_loss):.4f}')

            avg_loss = sum(epoch_loss) / len(epoch_loss)

        return avg_loss, dice_scores
