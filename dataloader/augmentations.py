import numpy as np
import torch


def DataTransform(sample, config):

    weak_aug = scaling(sample, config.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.max_seg), config.jitter_ratio)
    strong_aug2 = jitter(permutation(sample, max_segments=config.max_seg), config.jitter_ratio)

    return weak_aug, strong_aug, strong_aug2

def DataTransform_FD(sample, config):
    """Weak and strong augmentations in Frequency domain """
    weak_aug = remove_frequency(sample, pertub_ratio=0.1)
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_3 = remove_frequency(sample, pertub_ratio=0.1)
    aug_4 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    aug_F2 = aug_3 + aug_4
    # print(f'aug_F:{aug_F}')
    # print(f'aug_F2:{aug_F2}')
    return weak_aug, aug_F, aug_F2

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0.0):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix

