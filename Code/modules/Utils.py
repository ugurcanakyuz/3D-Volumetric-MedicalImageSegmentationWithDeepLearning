import glob

def create_patch_indexes(patch_counts, image_shape):
    px, py, pz = patch_counts
    x, y, z = image_shape
    ps_x, ps_y, ps_z = int(x / px), int(y / py), int(z / pz)  # patch sizes of each dimension

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

    files = glob.glob(path_data + "**/*.nii.gz", recursive=True)
    files = sorted(files)

    def pairwise(files):
        iterator = iter(files)

        return zip(iterator, iterator)

    paths = {}

    for image, mask in pairwise(files):
        paths[image.split('/')[1]] = [image, mask]

    return paths
