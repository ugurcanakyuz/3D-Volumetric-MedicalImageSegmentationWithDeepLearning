import torch
import torch.nn.functional as F
from modules.Utils import calculate_dice_score, create_onehot_mask


class Evaluator3D:
    """This class consist of evaluation method. Evaluate method calculates dice score per channel.

    Example:
        evaluator = Evaluator3D(model, patch_indexes, test_loader)
        avg_score = evaluator.evaluate(model)
    """

    def __init__(self, criterion, model, patch_indexes, val_loader):
        self.patch_indexes = patch_indexes
        self.val_loader = val_loader
        self.device = next(model.parameters()).device
        self.val_criterion = criterion

        out_channels = model.out.out_channels
        sample = next(iter(val_loader))
        shape = tuple(sample[0].shape[1:])
        self.output_shape = (val_loader.batch_size, out_channels, *shape)

    def evaluate(self, model):
        """Calculates dice score for each class.

        Parameters
        ----------
        model: torch.Model

        Returns
        -------
        avg_scores: float
            Average dice score for each class.
        """

        running_losses = []
        running_dice_scores = torch.zeros(self.output_shape[1]).to(self.device)

        with torch.no_grad():
            for j, (image, mask) in enumerate(self.val_loader):
                image = image.to(self.device)  # [bs,x,y,z]
                image = image.unsqueeze(1)  # [bs,1,x,y,z]

                mask = mask.to(self.device)  # [x,y,z]
                mask = mask.unsqueeze(1)  # [bs,1,x,y,z]

                for coors in self.patch_indexes:
                    [sx, sy, sz] = coors[0]
                    [ex, ey, ez] = coors[1]
                    patch_image = image[:, :, sx:ex, sy:ey, sz:ez]
                    patch_mask = mask[:, :, sx:ex, sy:ey, sz:ez]

                    unique, counts = torch.unique(mask[sx:ex, sy:ey, sz:ez], return_counts=True)

                    # if the label count below the 70 pixel don't train with it.
                    if torch.any(counts[1:] > 70):
                        output = model(patch_image)
                        val_loss = self.val_criterion(output, patch_mask)
                        running_losses.append(val_loss.item())

                        one_hot_mask = create_onehot_mask(output.shape, patch_mask)
                        output = F.softmax(output, dim=1)

                        scores = calculate_dice_score(output, one_hot_mask)
                        running_dice_scores += scores

            avg_loss = sum(running_losses) / len(running_losses)
            avg_scores = running_dice_scores / len(running_losses)

        return avg_loss, avg_scores


class Evaluator2D:
    """This class consist of evaluation method. Evaluate method calculates dice score per channel.

    Example:
        evaluator = Evaluator2D(model, test_loader)
        avg_score = evaluator.evaluate(model)
    """

    def __init__(self, criterion, model, val_loader):
        self.val_loader = val_loader
        self.bs_2d = 16  # Batch size of 2D slices.
        self.device = next(model.parameters()).device
        self.val_criterion = criterion

        out_channels = model.out.out_channels
        sample = next(iter(val_loader))
        shape = tuple(sample[0].shape[1:])
        self.output_shape = (val_loader.batch_size, out_channels, *shape)

    def evaluate(self, model):
        """Calculates dice score for each class.

        Parameters
        ----------
        model: torch.Model

        Returns
        -------
        avg_scores: float
            Average dice score for each class.
        """

        running_losses = []
        running_dice_scores = torch.zeros(self.output_shape[1]).to(self.device)
        count_forward = 0

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

                    output = model(slice_image)
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
