import torch
import torch.nn.functional as F
import torchio as tio
import tqdm

from modules.Utils import calculate_dice_score, create_onehot_mask


class Evaluator2D:
    """This class consist of evaluation method. Evaluate method calculates dice score per channel.

    Example:
        evaluator = Evaluator2D(model, test_loader)
        avg_score = evaluator.evaluate(model)
    """

    def __init__(self, criterion, model, val_loader):
        self.model = model
        self.val_loader = val_loader
        self.bs_2d = 16  # Batch size of 2D slices.
        self.device = next(model.parameters()).device
        self.val_criterion = criterion

        out_channels = model.out.out_channels
        sample = next(iter(val_loader))
        shape = tuple(sample[0].shape[1:])
        self.output_shape = (val_loader.batch_size, out_channels, *shape)

    def evaluate(self):
        """Calculates dice score for each class.

        Parameters
        ----------
        None.

        Returns
        -------
        avg_scores: float
            Average dice score for each class.
        """

        running_losses = []
        running_dice_scores = torch.zeros(self.output_shape[1]).to(self.device)
        count_forward = 0

        self.model.eval()
        with torch.no_grad():
            for j, (image, mask) in enumerate(self.val_loader):
                image = image.to(self.device)  # [bs,x,y,z]
                mask = mask.to(self.device)  # [bs,x,y,z] Most likely bs=1.
                mask = torch.unsqueeze(mask, 1)  # [bs, 1, x,y,z]
                output_mask = torch.Tensor().to(self.device)

                for slice_ix in range(0, image.shape[1], self.bs_2d):
                    start = slice_ix
                    stop = slice_ix + self.bs_2d

                    if stop > image.shape[1]:
                        stop = image.shape[1]

                    slice_image = image[:, start:stop]
                    slice_image = slice_image.view(-1, 1, 256, 256)

                    output = self.model(slice_image)
                    output_mask = torch.cat((output_mask, output))

                output_mask = output_mask.view(mask.shape[0], *output_mask.shape)  # [bs_3d, bs_2d, n_c, x, y]
                output_mask = output_mask.permute(0, 2, 1, 3, 4).contiguous()   # [bs_3d, n_c, x(bs_2d), y, z]

                val_loss = self.val_criterion(output_mask, mask)
                running_losses.append(val_loss.item())

                # To calculate DS, convert raw model outputs to softmax outputs.
                output_mask = F.softmax(output_mask, dim=1)
                one_hot_mask = create_onehot_mask(output_mask.shape, mask)

                scores = calculate_dice_score(output_mask, one_hot_mask)
                running_dice_scores += scores
                count_forward += 1

            avg_loss = sum(running_losses) / len(running_losses)
            avg_scores = running_dice_scores / count_forward

        return avg_loss, avg_scores


class Evaluator3D:
    """This class consist of evaluation method. Evaluate method calculates dice score per channel.

    Example:
        evaluator = Evaluator3D(model, patch_indexes, test_loader)
        avg_score = evaluator.evaluate(model)
    """

    def __init__(self, criterion, model, patch_size, val_loader):
        self.model = model
        self.patch_size = patch_size
        self.val_loader = val_loader
        self.device = next(model.parameters()).device
        self.val_criterion = criterion

        out_channels = model.out.out_channels
        sample = next(iter(val_loader))
        shape = tuple(sample[0].shape[1:])
        self.output_shape = (val_loader.batch_size, out_channels, *shape)

    def evaluate(self):
        """Calculates dice score for each class.

        Parameters
        ----------
        None.

        Returns
        -------
        avg_scores: float
            Average dice score for each class.
        """

        avg_val_loss = None
        epoch_loss = []
        overlap_mode_ = 'crop'
        running_losses = []
        running_dice_scores = []

        sampler = tio.data.GridSampler(subject=None, patch_size=self.patch_size)

        prog_bar = tqdm.tqdm(enumerate(self.val_loader),
                             total=int(len(self.val_loader) / self.val_loader.batch_size))
        prog_bar.set_description(f"Validation ")
        prog_bar.set_postfix_str(f'Loss: {avg_val_loss}')

        self.model.eval()
        with torch.no_grad():
            for i, (image, mask) in prog_bar:
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image),
                    mask=tio.LabelMap(tensor=mask),
                )
                sampler.subject = subject
                aggregator = tio.data.GridAggregator(sampler, overlap_mode=overlap_mode_)

                for j, patch in enumerate(sampler(subject)):
                    patch_image = patch["image"].data.unsqueeze(1).to(self.device)  # [bs,1,x,y,z]

                    output = self.model(patch_image)
                    aggregator.add_batch(output, patch["location"].unsqueeze(0))

                output = aggregator.get_output_tensor().unsqueeze(0)
                mask = mask.unsqueeze(0)

                # Validation loss calculated on the aggregated so whole predicted mask.
                # Criterion accepts raw logits so softmax has not been applied to predicted mask yet.
                val_loss = self.val_criterion(output, mask)
                running_losses.append(val_loss.item())

                # To calculate Dice Score get softmax applied predicted mask.
                pred_mask = F.softmax(output, dim=1)
                # Create one hot encoded mask of original mask.
                one_hot_mask = create_onehot_mask(pred_mask.shape, mask)
                # Convert class probabilities to actual class labels.
                pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
                # Create one hot encoded mask of predicted mask.
                pred_mask = create_onehot_mask(one_hot_mask.shape, pred_mask)

                scores = calculate_dice_score(pred_mask, one_hot_mask)
                running_dice_scores.append(scores)

                avg_loss = sum(running_losses) / len(running_losses)
                running_losses = []
                epoch_loss.append(avg_loss)
                prog_bar.set_postfix_str(f'Loss: {sum(epoch_loss) / len(epoch_loss):.4f}')

            avg_loss = sum(epoch_loss) / len(epoch_loss)
            avg_scores = sum(running_dice_scores) / len(running_dice_scores)

        return avg_loss, avg_scores

    def predict(self, image):
        self.model.eval()

        overlap_mode_ = 'crop'

        sampler = tio.data.GridSampler(subject=None, patch_size=self.patch_size)

        with torch.no_grad():
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
            )

            sampler.subject = subject
            aggregator = tio.data.GridAggregator(sampler, overlap_mode=overlap_mode_)

            for j, patch in enumerate(sampler(subject)):
                patch_image = patch["image"].data.unsqueeze(1).to(self.device)  # [bs,1,x,y,z]
                output = self.model(patch_image)
                aggregator.add_batch(output, patch["location"].unsqueeze(0))

            output = aggregator.get_output_tensor().unsqueeze(0)

        return output
