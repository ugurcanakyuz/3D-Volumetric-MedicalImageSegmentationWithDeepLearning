import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn.functional as F
import torchio as tio
import tqdm

from torch.utils.data import DataLoader
from utils.Utils import calculate_dice_score, create_onehot_mask


class Evaluator3D:
    """This class consist of evaluation method. Evaluate method calculates dice score per channel.

    Example:
        evaluator = Evaluator3D(model, model, val_loader)
        avg_score = evaluator.evaluate()
    """

    def __init__(self, criterion, model, patch_size, val_loader, out_channels=8):
        """Creates 3D evaluator object.

        Parameters
        ----------
        criterion: modules.LossFunctions
            To calculate loss value of 3D Data.
        model: modules.UNet
            3D UNet model implemented in UNet.
        val_loader: DataLoader
            3D Data loader.
        out_channels: int

        Returns
        -------
        None
        """

        self.cm = None
        self.model = model
        self.patch_size = patch_size
        self.val_loader = val_loader
        self.device = next(model.parameters()).device
        self.val_criterion = criterion

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

    def calculate_cm(self):
        """Calculate confusion matrix for each class.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        overlap_mode_ = 'average'
        prog_bar = tqdm.tqdm(enumerate(self.val_loader),
                             total=int(len(self.val_loader) / self.val_loader.batch_size))
        prog_bar.set_description(f"Validation ")

        cms = []

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
                # Convert class probabilities to actual class labels.
                pred_mask = torch.argmax(output, dim=1, keepdim=True)

                cm = confusion_matrix(mask.ravel(), pred_mask.ravel())
                cms.append(cm)

        self.cm = sum(cms)

        return self.cm

    def plot_confusion_matrix(self, lang=None):
        """Plot confusion matrix for each class.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        if lang=='tr':
            labels = ['Arka Plan', 'eCSF', 'Gri Cevher', 'Beyaz Cevher',
                      'Ventriküller', 'Beyincik', 'TaPu',
                      'Beyin Sapı']
        else:
            labels = ['Background', 'eCSF', 'Gray Matter', 'White Matter',
                      'Ventricles', 'Cerrebilium', 'Deep Gray Matter',
                      'Brain Stem']

        assert self.cm is not None, "Calculate confusion matrix first. Call <EvaluatorObject>.calculate_cm"
        # Normalize confusion matrix
        cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 6))
        fx = sns.heatmap(cm, annot=True, fmt='.2f', cmap='GnBu')
        if lang=='tr':
            fx.set_title('Hata Matrisi \n')
            fx.set_xlabel('\n Tahmini Değerler\n')
            fx.set_ylabel('Gerçek Değerler\n')
        else:
            fx.set_title('Confusion Matrix \n')
            fx.set_xlabel('\n Predicted Values\n')
            fx.set_ylabel('Actual Values\n')
        fx.xaxis.set_ticklabels(labels, rotation=45, ha="right")
        fx.yaxis.set_ticklabels(labels, rotation=45, ha="right")
        plt.show()
