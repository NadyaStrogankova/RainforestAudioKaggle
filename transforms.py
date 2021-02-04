import torch
import torchaudio


def random_power(images, power=1.5, c=0.7):
    images = images ** (torch.rand(1, device="cuda:0") * power + c)
    # print(images.dtype)
    return images


def normalize_mel(melspec):
    melspec -= melspec.min()
    melspec /= melspec.max()
    # print(melspec)
    return melspec


def normalize_image(melspec):
    # normalize
    m = melspec.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    melspec -= m
    s = melspec.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    # s = torch.maximum(s, torch.ones_like(s)*1e-7)
    melspec /= s
    # print(melspec)
    return melspec


def normalize_channel(melspec):
    # normalize
    m = melspec.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    melspec -= m
    s = melspec.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    # s = torch.maximum(s, torch.ones_like(s)*1e-7)
    melspec /= s
    # print(melspec)
    return melspec


class Normalize_channel_1(Transform):
    def __init__(self):
        split_idx = None

    def encodes(self, melspec: AudioSpectrogram) -> AudioSpectrogram:
        return normalize_image(melspec)


class Normalize_channel_2(Transform):
    def __init__(self):
        split_idx = None

    def encodes(self, melspec: AudioSpectrogram) -> AudioSpectrogram:
        return normalize_image(melspec)


class Normalize_channel_3(Transform):
    def __init__(self):
        split_idx = None

    def encodes(self, melspec: AudioSpectrogram) -> AudioSpectrogram:
        return normalize_image(melspec)


class PowerSpec(Transform):
    """
    Transform для возведения спектрограммы в степень.
    """

    def __init__(self, power=2, c=0.7):
        self.power = power
        self.c = c
        split_idx = None

    def encodes(self, melspec: AudioSpectrogram) -> AudioSpectrogram:
        mel = random_power(melspec, self.power, self.c)
        # print(melspec, mel, mel.size())
        return mel


class LowerUpperFreq(Transform):
    """
    Transform для понижения верхних частот.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        split_idx = None

    def encodes(self, images: AudioSpectrogram) -> AudioSpectrogram:
        if torch.rand(1) < 0.5:
            images = images - images.min(dim=2, keepdim=True)[0]
            r = random.randint(self.cfg.n_mels // 2, self.cfg.n_mels)
            x = random.random() / 2
            # print(r, x, torch.zeros(self.cfg.n_mels-r)-x+1)
            pink_noise = torch.cat((1 - torch.arange(r, device="cuda:0") * x / r,
                                    torch.ones(self.cfg.n_mels - r, device="cuda:0") - x)).T

            # pink_noise = np.array([np.concatenate((1-np.arange(r)*x/r,np.zeros(self.hp.n_mels-r)-x+1))]).T
            # print(images.size(), pink_noise.size(), pink_noise)
            # print(pink_noise, images)
            images = images.movedim(3, 2)

            images = images * pink_noise
            # images = images/(images.max(dim =2, keepdim=True)[0]
            images = images.movedim(3, 2)
            # print(images.shape, images)
        return images


class RowNoise(Transform):
    """
    Transform полосового шума.
    """

    def __init__(self, level_noise, cfg):
        self.cfg = cfg
        self.level_noise = level_noise
        self.split_idx = 0

    def encodes(self, images: AudioSpectrogram) -> AudioSpectrogram:
        if torch.rand(1) < 0.9:
            a = int(torch.rand(1, device="cuda:0") * self.cfg.n_mels // 2)
            b = int(torch.min(torch.rand(1, device="cuda:0") * self.cfg.n_mels, a + 20))
            images[a:b, :] = images[a:b, :] + (torch.rand((b - a, images.size()[-1]),
                                                          device="cuda:0") + 9.0) * 0.05 * images.mean() * self.level_noise * (
                                     torch.rand(1, device="cuda:0") + 0.3)

        return images


class PinkNoise(Transform):
    """
    Transform для розового шума.
    """

    def __init__(self, level_noise, cfg):
        self.level_noise = level_noise
        self.cfg = cfg
        self.split_idx = 0

    def encodes(self, images: AudioSpectrogram) -> AudioSpectrogram:
        if torch.rand(1) < 0.9:
            r = random.randint(1, self.cfg.n_mels)
            # print(r, x, torch.zeros(self.cfg.n_mels-r)-x+1)
            pink_noise = torch.cat((1 - torch.arange(r, device="cuda:0") / r,
                                    torch.zeros(self.cfg.n_mels - r, device="cuda:0"))).T

            # pink_noise = np.array([np.concatenate((1-np.arange(r)*x/r,np.zeros(self.hp.n_mels-r)-x+1))]).T
            # print(images.size(), pink_noise.size(), pink_noise)

            images = images + (torch.rand((self.cfg.n_mels, images.size()[-1]),
                                          device="cuda:0") + 9.0) * 2 * images.mean() * self.level_noise * (
                             torch.rand(1, device="cuda:0") + 0.3)
            # print(pink_noise, images)
            # print(images.shape, images)
        return images


class WhiteNoise(Transform):
    """
    Transform для розового шума.
    """

    def __init__(self, level_noise, cfg):
        self.level_noise = level_noise
        self.cfg = cfg
        self.split_idx = 0

    def encodes(self, images: AudioSpectrogram) -> AudioSpectrogram:
        if torch.rand(1) < 0.9:
            images = images + (torch.rand((self.cfg.n_mels, images.size()[-1]),
                                          device="cuda:0") + 9.0) * images.mean() * self.level_noise * (
                             torch.rand(1, device="cuda:0") + 0.3)
            # print(images.shape, images)
        return images


class Mono2Color(Transform):
    """
    Transform для создания цветной картинки.
    """

    def __init__(self):
        split_idx = None

    def encodes(self, images: AudioSpectrogram) -> AudioSpectrogram:
        delta = torchaudio.functional.compute_deltas(images)
        delta2 = torchaudio.functional.compute_deltas(delta)
        colored_image = torch.cat((images, delta, delta2), dim=1)
        # print(colored_image.shape)
        return colored_image


class Mask(Transform):
    """
    Transform для рандомизации MaskFreq и MaskTime.
    """

    def __init__(self):
        split_idx = None

    def encodes(self, images: AudioSpectrogram) -> AudioSpectrogram:
        rnd = torch.rand(1)
        # print(images[0].shape, images.dtype, rnd, int(rnd*8))
        # if isinstance(images, AudioSpectrogram):
        if rnd < 0.25:
            images.data = MaskFreq(num_masks=int(rnd * 8), size=20)(images)
            images.data = MaskTime(num_masks=int(rnd * 8), size=16)(images)
        elif rnd < 0.5:
            images.data = MaskFreq(num_masks=int(rnd * 4), size=20)(images)
        elif rnd < 0.75:
            images.data = MaskTime(num_masks=int(rnd * 4), size=16)(images)

        return images
