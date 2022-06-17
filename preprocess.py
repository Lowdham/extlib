import numpy as np
import torch
from extlib.libclass import *
from extlib.conf import *
from torch.autograd import Variable


std = np.array([0.5, 0.5, 0.5])
mean = np.array([1, 1, 1])


# We assume the height and width are already adjusted that can be divided evenly.
def divide_image(height, width, size):
    vertical_tiles_num = int(height / size)
    horizontal_tile_num = int(width / size)

    # Prepare the tiles.
    up_tiles = []
    down_tiles = []
    left_tiles = []
    right_tiles = []

    # Upper tiles.
    for i in range(0, horizontal_tile_num):
        up_tiles.append(tile(point(i * size, 0), point((i + 1) * size, size)))

    # Down tiles.
    for i in range(0, horizontal_tile_num):
        down_tiles.append(tile(point(i * size, (vertical_tiles_num - 1) * size), point((i + 1) * size, height)))

    # Left tiles.
    for i in range(0, vertical_tiles_num):
        left_tiles.append(tile(point(0, i * size), point(size, (i + 1) * size)))

    # Right tiles.
    for i in range(0, vertical_tiles_num):
        right_tiles.append(tile(point((horizontal_tile_num - 1) * size, i * size), point(width, (i + 1) * size)))

    return {'up_tiles': up_tiles,
            'down_tiles': down_tiles,
            'left_tiles': left_tiles,
            'right_tiles': right_tiles}


# Cut the images according to the box
def cut_image(raw_image, boxes):
    tiles = []
    for target in boxes:
        x1, y1, x2, y2 = target.get_box()
        tiles.append(img_cut(raw_image, (x1, y1, x2, y2)))
    return tiles


# Convert the tiles into the numpy array.
def img2arr(tiles):
    converted = np.asarray(tiles).astype('f').transpose(2, 0, 1)
    return converted


# Convert the tiles into the numpy array.
def imgs2arr(tiles):
    result = []
    for unconverted in tiles:
        arr = img2arr(unconverted)
        result.append(arr)

    return result


# Convert the tiles into the numpy array.
def arr2img(tiles_arr):
    img = Image.fromarray(tiles_arr.transpose(1, 2, 0).astype('uint8'))
    return img


# Convert the tiles into the numpy array.
def arrs2img(tiles_arrs):
    result = []
    for unconverted in tiles_arrs:
        img = arr2img(unconverted)
        result.append(img)

    return result


# Normalize (RGB: 0 ~ 255)
def normalize_image(raw):
    return raw / 127.5 - 1.0


# Unnormalize
def unnormalize_image(img):
    return (img + 1) * 127.5 + 0.5


# Normalize (RGB: 0 ~ 255)
def normalize_images(raw_tiles):
    tiles = []
    for unnormalized in raw_tiles:
        tiles.append(unnormalized/127.5-1.0)
    return tiles


# Unnormalize
def unnormalize_images(tiles):
    result = []
    for normalized in tiles:
        result.append((normalized + 1) * 127.5 + 0.5)
    return result


# Reverse the mode
def reverse_mode(mode):
    if mode == 'up':
        return "down"
    if mode == "down":
        return "up"
    if mode == "left":
        return "right"
    if mode == "right":
        return "left"


# Recompute the boxes
def shift_box(boxes, box_size, mode):
    if mode == 'up':
        last_tile = boxes[len(boxes) - 1]
        ul_pnt = last_tile.ul_pnt
        dr_pnt = last_tile.dr_pnt
        boxes.append(tile(point(ul_pnt.x + box_size, ul_pnt.y), point(dr_pnt.x + box_size, dr_pnt.y)))
        boxes.append(tile(point(ul_pnt.x + 2 * box_size, ul_pnt.y), point(dr_pnt.x + 2 * box_size, dr_pnt.y)))
        return boxes

    if mode == 'down':
        for unshift in boxes:
            unshift.ul_pnt.y = unshift.ul_pnt.y + 2 * box_size
            unshift.dr_pnt.y = unshift.dr_pnt.y + 2 * box_size

        last_tile = boxes[len(boxes) - 1]
        ul_pnt = last_tile.ul_pnt
        dr_pnt = last_tile.dr_pnt
        boxes.append(tile(point(ul_pnt.x + box_size, ul_pnt.y), point(dr_pnt.x + box_size, dr_pnt.y)))
        boxes.append(tile(point(ul_pnt.x + 2 * box_size, ul_pnt.y), point(dr_pnt.x + 2 * box_size, dr_pnt.y)))
        return boxes

    if mode == 'left':
        last_tile = boxes[len(boxes) - 1]
        ul_pnt = last_tile.ul_pnt
        dr_pnt = last_tile.dr_pnt
        boxes.append(tile(point(ul_pnt.x, ul_pnt.y + box_size), point(dr_pnt.x, dr_pnt.y + box_size)))
        boxes.append(tile(point(ul_pnt.x, ul_pnt.y + 2 * box_size), point(dr_pnt.x, dr_pnt.y + 2 * box_size)))
        return boxes

    if mode == 'right':
        for unshift in boxes:
            unshift.ul_pnt.x = unshift.ul_pnt.x + 2 * box_size
            unshift.dr_pnt.x = unshift.dr_pnt.x + 2 * box_size

        last_tile = boxes[len(boxes) - 1]
        ul_pnt = last_tile.ul_pnt
        dr_pnt = last_tile.dr_pnt
        boxes.append(tile(point(ul_pnt.x, ul_pnt.y + box_size), point(dr_pnt.x, dr_pnt.y + box_size)))
        boxes.append(tile(point(ul_pnt.x, ul_pnt.y + 2 * box_size), point(dr_pnt.x, dr_pnt.y + 2 * box_size)))
        return boxes


# Paste to the new image according to the box
def paste_img(target, tiles, boxes):
    if len(tiles) != len(boxes):
        raise Exception("The length of tiles doesn't match the length of its boxes.")

    fragments = arrs2img(tiles)

    for idx in range(0, len(fragments)):
        fragment = fragments[idx]
        box = boxes[idx]
        target.paste(fragment, (box.ul_pnt.x, box.ul_pnt.y, box.dr_pnt.x, box.dr_pnt.y))


# Create the mask according to the mode and ratio (default implement is based on normal RGB image)
def create_mask(height, width, mode, ratio):
    mask = np.zeros((height, width))

    if mode == 'up':
        edge = int(height * ratio)
        mask[edge:, :] = 1

    if mode == 'down':
        edge = int(height * (1 - ratio))
        mask[:edge, :] = 1

    if mode == 'left':
        edge = int(width * ratio)
        mask[:, edge:] = 1

    if mode == 'right':
        edge = int(width * (1 - ratio))
        mask[:, :edge] = 1

    return mask


# Auto cut and cat the raw image to a new image.
def cut_and_cat(raw_img, mode, ratio):
    new_img = np.zeros(raw_img.shape)
    height = raw_img.shape[len(raw_img.shape) - 2]
    width = raw_img.shape[len(raw_img.shape) - 1]

    if mode == 'up':
        edge = int(height * ratio)
        new_img[:, edge:, :] = raw_img[:, :height - edge, :]

    if mode == 'down':
        edge = int(height * ratio)
        new_img[:, :height - edge, :] = raw_img[:, edge:, :]

    if mode == 'left':
        edge = int(width * ratio)
        new_img[:, :, edge:] = raw_img[:, :, :width - edge]

    if mode == 'right':
        edge = int(width * ratio)
        new_img[:, :, :width - edge] = raw_img[:, :, edge:]

    return new_img


# Mask on the tiles.
def mask_tile(tile_img, mode, ratio, device):
    height = tile_img.shape[len(tile_img.shape) - 2]
    width = tile_img.shape[len(tile_img.shape) - 1]

    # Create the mask.
    mask = create_mask(height, width, mode, ratio)
    mask = 1 - mask
    mask = mask[np.newaxis, :, :]

    # Cut and cat.
    img = cut_and_cat(tile_img, mode, ratio)
    img = tile_img * (1 - mask)

    # Torch.
    img = torch.from_numpy(img)
    mask = torch.from_numpy(mask)
    mask_one = torch.ones((height, width), dtype=torch.float64)

    clip = torch.cat([img, mask_one[None, :, :], mask]).float()
    mask = mask.float().to(device)

    return Variable(clip).to(device).unsqueeze(0), \
           Variable(img).to(device),\
           Variable(mask).to(device)
