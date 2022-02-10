import glob
import torch


def create_patch_indexes(image_shape, patch_sizes):
    """Creates image patch coordinates for 3D dimension. Image dimensions must be divisible by patch size without
    remainder.

    Parameters
    ----------
    image_shape: tuple
    patch_sizes: tuple

    Returns
    -------
    patches: list
    """

    x, y, z = image_shape
    ps_x, ps_y, ps_z = patch_sizes
    assert x % ps_x == 0, "First dimension of the image must be divisible by patch size without remainder."
    assert y % ps_y == 0, "Second dimension of the image must be divisible by patch size without remainder."
    assert z % ps_z == 0, "Third dimension of the image must be divisible by patch size without remainder."
    px, py, pz = int(x / ps_x), int(y / ps_y), int(z / ps_z)

    sx, sy, sz = 0, 0, 0  # starting points

    patches = []

    for i in range(px):
        for j in range(py):
            for u in range(pz):
                patches.append([[sx, sy, sz], [sx + ps_x, sy + ps_y, sz + ps_z]])
                sz += ps_z
            sz = 0
            sy += ps_y
        sy = 0
        sx += ps_x

    return patches


def get_file_names(path_data):
    """List the files in sub directories.

    Parameters
    ----------
    path_data: str
        Path of the data folder.

    Returns
    -------
    paths: dict
        {'sub_name': [image_name, mask_name]}
    """

    files = glob.glob(path_data + "/**/*.nii.gz", recursive=True)
    files = sorted(files)

    def pairwise(files):
        iterator = iter(files)

        return zip(iterator, iterator)

    paths = {}

    for image, mask in pairwise(files):
        paths[image.split('/')[1]] = [image, mask]

    return paths


def create_onehot_mask(pred_shape, mask, device):
    """Creates onehot mask for multidimensional mask.

    Parameters
    ----------
    pred_shape: tuple
        (bs,number_of_classes,x,y)
                    or
        (bs,number_of_classes,x,y,z)
    mask: torch.tensor
        (bs,1,x,y)
            or
        (bs,1,x,y,z)
    device: str
        'cuda' or 'cpu'...
    Returns
    -------
    mask_onehot: torch.tensor
        [bs,number_of_classes,x,y,z]
    """

    mask_onehot = torch.zeros(pred_shape).to(device)
    mask = mask.long()
    mask_onehot.scatter_(1, mask, 1)

    return mask_onehot


def normal_weight_init(m):
    """Initialize model weights with uniform distribution.

    Parameters
    ----------
    m: torch.model

    Returns
    -------
    None
    """

    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
