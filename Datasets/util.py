import numpy as np

def split2list(plyfiles, split, default_split= 0.9):
    if isinstance(split, str):
            with open(split) as f:
                split_values=[x.strip() == '1' for x in f.readlines()]
            assert(len(plyfiles) == len(split_values))
    elif isinstance(split, float):
            split_values = np.random.uniform(0,1,len(plyfiles)) < split
    else:
        split_values = np.random.uniform(0,1,len(plyfiles)) < default_split
    train_samples = [sample for sample, split in zip(plyfiles, split_values) if split]
    test_samples = [sample for sample, split in zip(plyfiles, split_values) if not split]
    return train_samples, test_samples

def load_ply(file_name, with_faces=False, with_color=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val