def create_patch_indexes(patch_counts, image_shape):
    px, py, pz = patch_counts
    x, y, z = image_size
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