from PIL import Image

# GPU Device
gpu_device = 0


# Image open
def img_open(img_path):
    return Image.open(img_path)


# Image cut
def img_cut(raw_image, box):
    return raw_image.crop(box)
