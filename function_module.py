from extlib.preprocess import *
from abc import abstractmethod


class MaskModule:
    @abstractmethod
    def mask(self, tile_img, mode, ratio, device):
        pass


class EdgeExtendedMaskModule(MaskModule):
    def mask(self, tile_img, mode, ratio, device):
        height = tile_img.shape[len(tile_img.shape) - 2]
        width = tile_img.shape[len(tile_img.shape) - 1]

        # Create the mask.
        mask = create_mask(height, width, mode, ratio)
        mask = 1 - mask
        mask = mask[np.newaxis, :, :]

        # Cut and cat.
        img = cut_and_cat(tile_img, mode, ratio)

        # Torch.
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        mask_one = torch.ones((height, width), dtype=torch.float64)

        clip = torch.cat([img, mask_one[None, :, :], mask]).float()
        mask = mask.float().to(device)

        return Variable(clip).to(device).unsqueeze(0), \
               Variable(img).to(device), \
               Variable(mask).to(device)
