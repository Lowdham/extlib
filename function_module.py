from extlib.preprocess import *
from extlib.preprocess import _rot2right, _reverse_rot2right
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

        # To pytorch data structure.
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        mask_one = torch.ones((height, width), dtype=torch.float64)

        clip = torch.cat([img, mask_one[None, :, :], mask]).float()
        mask = mask.float().to(device)

        return Variable(clip).to(device).unsqueeze(0), \
               Variable(img).to(device), \
               Variable(mask).to(device)


class QuarterExtendedMaskModule(MaskModule):
    def mask(self, tile_img, mode, ratio, device):
        height = tile_img.shape[len(tile_img.shape) - 2]
        width = tile_img.shape[len(tile_img.shape) - 1]

        # Create the mask.
        mask = create_quarter_mask(height, width, mode, ratio)
        mask = 1 - mask
        mask = mask[np.newaxis, :, :]

        # To pytorch data structure.
        img = torch.from_numpy(tile_img)
        mask = torch.from_numpy(mask)
        mask_one = torch.ones((height, width), dtype=torch.float64)

        clip = torch.cat([img, mask_one[None, :, :], mask]).float()
        mask = mask.float().to(device)

        return Variable(clip).to(device).unsqueeze(0), \
               Variable(img).to(device), \
               Variable(mask).to(device)


class UpScanningModule:
    def __init__(self, tile_size, width, height):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.step = 64
        self.current_point = (self.step, height - self.step)

    def get_current_box(self):
        # Upward.
        x, y = self.current_point
        box = (x, y - self.tile_size, x + self.tile_size, y)
        return box

    def next(self):
        x, y = self.current_point
        if y == self.tile_size:
            if x == self.width - self.tile_size:
                return False

            self.current_point = (x + self.step, self.height - self.step)
        else:
            self.current_point = (x, y - self.step)

        return True

    def rot(self, arr):
        return arr

    def rrot(self, arr):
        return arr


class DownScanningModule:
    def __init__(self, tile_size, width, height):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.step = 64
        self.current_point = (width - self.step, self.step)

    def get_current_box(self):
        # Downward.
        x, y = self.current_point
        box = (x - self.tile_size, y, x, y + self.tile_size)
        return box

    def next(self):
        x, y = self.current_point
        if y == self.tile_size:
            if x == self.tile_size:
                return False

            self.current_point = (x - self.step, self.step)
        else:
            self.current_point = (x, y + self.step)

        return True

    def rot(self, arr):
        return _rot2right('down', arr)

    def rrot(self, arr):
        return _reverse_rot2right('down', arr)


class LeftScanningModule:
    def __init__(self, tile_size, width, height):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.step = 64
        self.current_point = (width - self.step, height - self.step)

    def get_current_box(self):
        # Leftward.
        x, y = self.current_point
        box = (x - self.tile_size, y - self.tile_size, x, y)
        return box

    def next(self):
        x, y = self.current_point
        if x == self.tile_size:
            if y == self.tile_size:
                return False

            self.current_point = (self.width - self.step, y - self.step)
        else:
            self.current_point = (x - self.step, y)

        return True

    def rot(self, arr):
        return _rot2right('left', arr)

    def rrot(self, arr):
        return _reverse_rot2right('left', arr)


class RightScanningModule:
    def __init__(self, tile_size, width, height):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.step = 64
        self.current_point = (self.step, self.step)

    def get_current_box(self):
        # Rightward.
        x, y = self.current_point
        box = (x, y, x + self.tile_size, y + self.tile_size)
        return box

    def next(self):
        x, y = self.current_point
        if x == self.tile_size:
            if y == self.width - self.tile_size:
                return False

            self.current_point = (self.step, y + self.step)
        else:
            self.current_point = (x + self.step, y)

        return True

    def rot(self, arr):
        return _rot2right('right', arr)

    def rrot(self, arr):
        return _reverse_rot2right('right', arr)


class QuarterScanningModule:
    def __init__(self, workspace, tile_size, mode, width, height):
        self.workspace = workspace
        self.mode = mode
        self.step = 64
        if mode == 'up':
            self.scanning_module = UpScanningModule(tile_size, width, height)
        if mode == 'down':
            self.scanning_module = DownScanningModule(tile_size, width, height)
        if mode == 'left':
            self.scanning_module = LeftScanningModule(tile_size, width, height)
        if mode == 'right':
            self.scanning_module = RightScanningModule(tile_size, width, height)

    def extend(self, extend_once, ratio, device):
        box = self.scanning_module.get_current_box()
        fragment = self.scanning_module.rot(normalize_image(img2arr(img_cut(self.workspace, box))))
        img_tensor, img, mask = QuarterExtendedMaskModule().mask(fragment, "ur", ratio, device)
        generated_img = unnormalize_image(extend_once(img_tensor, img, mask).squeeze().cpu().numpy())

        # Paste back to the workspace.
        self.workspace.paste(arr2img(self.scanning_module.rrot(generated_img)), box)

        while self.scanning_module.next():
            box = self.scanning_module.get_current_box()
            fragment = self.scanning_module.rot(normalize_image(img2arr(img_cut(self.workspace, box))))
            img_tensor, img, mask = QuarterExtendedMaskModule().mask(fragment, "ur", ratio, device)
            generated_img = unnormalize_image(extend_once(img_tensor, img, mask).squeeze().cpu().numpy())

            # Paste back to the workspace.
            self.workspace.paste(arr2img(self.scanning_module.rrot(generated_img)), box)

    def get_workspace(self):
        return self.workspace
