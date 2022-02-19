import torch
import torch.nn.functional as F
from modules.Utils import calculate_dice_score, create_onehot_mask


class Evaluator3D:
    """This class consist of evaluation method. Evaluate method calculates dice score per channel.

    Example:
        evaluator = Evaluator3D(model, patch_indexes, test_loader)
        avg_score = evaluator.evaluate(model)
    """

    def __init__(self, model, patch_indexes, val_loader):
        self.patch_indexes = patch_indexes
        self.val_loader = val_loader

        self.device = next(model.parameters()).device

        out_channels = model.out.out_channels
        sample = next(iter(val_loader))
        shape = tuple(sample[0].shape[1:])
        self.output_shape = (val_loader.batch_size, out_channels, *shape)

    def evaluate(self, model):
        """Calculates dice score for each class.

        Parameters
        ----------
        model: torch.model

        Returns
        -------
        avg_scores: float
            Average dice score for each class.
        """

        running_dice_scores = torch.zeros(self.output_shape[1]).to(self.device)
        with torch.no_grad():
            for j, (image, mask) in enumerate(self.val_loader):
                image = image.to(self.device)  # [bs,x,y,z]
                image = image.view(self.output_shape[0], 1, *self.output_shape[2:])  # [bs,c,x,y,z]

                mask = mask.to(self.device)  # [x,y,z]
                mask = mask.view(self.output_shape[0], 1, *self.output_shape[2:])  # [bs,1,x,y,z]

                for coors in self.patch_indexes:
                    [sx, sy, sz] = coors[0]
                    [ex, ey, ez] = coors[1]
                    patch_image = image[:, :, sx:ex, sy:ey, sz:ez]
                    patch_mask = mask[:, :, sx:ex, sy:ey, sz:ez]

                    output = F.softmax(model(patch_image), dim=1)
                    one_hot_mask = create_onehot_mask(output.shape, patch_mask)

                    scores = calculate_dice_score(output, one_hot_mask)
                    running_dice_scores += scores

            avg_scores = running_dice_scores / len(self.val_loader) * len(self.patch_indexes)

        return avg_scores
