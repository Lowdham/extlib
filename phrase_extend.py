from extlib.preprocess import *


class TestGen:
    def __init__(self):
        self.color = 0
        self.layer = 0
        self.cnt = 0

    def __call__(self, arg):
        h = arg.shape[1]
        w = arg.shape[2]
        arr = np.zeros((3, h, w))
        arr[self.layer, :, :] = self.color

        self.cnt = self.cnt + 1
        if self.cnt == 1024:
            self.color = self.color + 1
            self.cnt = 0
        return torch.from_numpy(arr).to("cpu")


#
class ExtendEngine:
    def __init__(self, generator, mask_module, device):
        self.generator = generator
        self.mask_module = mask_module
        self.device = device

    def extend_once(self, image_tensor, img, mask):
        with torch.no_grad():
            gen = self.generator(image_tensor)
            result = gen * mask + img
            return result

    def extend_one_edge(self, tiles, mode, ratio):
        # Convert tiles into the numpy array.
        old_tiles = rot2right(mode, normalize_images(tiles))
        new_tiles = []

        for i in range(0, int(1 / ratio)):
            for unmask_tile in old_tiles:
                img_tensor, img, mask = self.mask_module.mask(unmask_tile, "right", ratio, self.device)
                generated_img = self.extend_once(img_tensor, img, mask)
                new_tiles.append(generated_img.squeeze().cpu().numpy())

            old_tiles = new_tiles
            new_tiles = []

        return reverse_rot2right(mode, unnormalize_images(old_tiles))

    def extend_corner(self, base_tile, mode, ratio):
        result_tile = rot2right(mode, [normalize_image(base_tile)])[0]
        for i in range(0, int(1 / ratio)):
            img_tensor, img, mask = self.mask_module.mask(result_tile, 'right', ratio, self.device)
            result_tile = self.extend_once(img_tensor, img, mask).squeeze().cpu().numpy()

        return reverse_rot2right(mode, [unnormalize_image(result_tile)])[0]

    def extend_all(self, meta_img_path, tile_size, ratio, times):
        raw_img = img_open(meta_img_path)
        width, height = raw_img.size

        #
        prev_img = raw_img
        new_img = prev_img

        # Prepare the images.
        tile_boxes = divide_image(height, width, tile_size)
        up_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['up_tiles']))
        down_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['down_tiles']))
        left_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['left_tiles']))
        right_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['right_tiles']))

        # Extend.
        for i in range(0, times):

            # First, extend all edges.
            up_tile_arrs = self.extend_one_edge(up_tile_arrs, 'up', ratio)
            down_tile_arrs = self.extend_one_edge(down_tile_arrs, 'down', ratio)
            left_tile_arrs = self.extend_one_edge(left_tile_arrs, 'left', ratio)
            right_tile_arrs = self.extend_one_edge(right_tile_arrs, 'right', ratio)

            # Second, extend the corner.
            up_left_corner = self.extend_corner(up_tile_arrs[0], 'left', ratio)
            up_right_corner = self.extend_corner(up_tile_arrs[len(up_tile_arrs) - 1], 'right', ratio)
            down_left_corner = self.extend_corner(down_tile_arrs[0], 'left', ratio)
            down_right_corner = self.extend_corner(down_tile_arrs[len(down_tile_arrs) - 1], 'right', ratio)

            # Smooth.

            # Update the tile arrays.
            up_tile_arrs.insert(0, up_left_corner)
            up_tile_arrs.append(up_right_corner)
            right_tile_arrs.insert(0, up_right_corner)
            right_tile_arrs.append(down_right_corner)
            left_tile_arrs.insert(0, up_left_corner)
            left_tile_arrs.append(down_left_corner)
            down_tile_arrs.insert(0, down_left_corner)
            down_tile_arrs.append(down_right_corner)

            # Recompute the boxes
            tile_boxes = {'up_tiles': shift_box(tile_boxes['up_tiles'], tile_size, 'up'),
                          'down_tiles': shift_box(tile_boxes['down_tiles'], tile_size, 'down'),
                          'left_tiles': shift_box(tile_boxes['left_tiles'], tile_size, 'left'),
                          'right_tiles': shift_box(tile_boxes['right_tiles'], tile_size, 'right')}

            # Merge into a new image.
            prev_width, prev_height = prev_img.size
            prev_img_box = (tile_boxes['up_tiles'][0].dr_pnt.x,
                            tile_boxes['up_tiles'][0].dr_pnt.y,
                            tile_boxes['up_tiles'][0].dr_pnt.x + prev_width,
                            tile_boxes['up_tiles'][0].dr_pnt.y + prev_height)

            new_img = Image.new(mode='RGB', size=(prev_height + 2 * tile_size, prev_width + 2 * tile_size))
            paste_img(new_img, up_tile_arrs, tile_boxes['up_tiles'])
            paste_img(new_img, down_tile_arrs, tile_boxes['down_tiles'])
            paste_img(new_img, left_tile_arrs, tile_boxes['left_tiles'])
            paste_img(new_img, right_tile_arrs, tile_boxes['right_tiles'])
            new_img.paste(prev_img, prev_img_box)

            prev_img = new_img

            print("Iteration " + str(i) + " Complete.")

        return new_img

    def extend_all_v2(self, meta_img_path, tile_size, ratio, times):
        raw_img = img_open(meta_img_path)
        width, height = raw_img.size

        #
        prev_img = raw_img
        new_img = prev_img

        # Prepare the images.
        tile_boxes = divide_image(height, width, tile_size)
        left_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['left_tiles']))
        right_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['right_tiles']))

        # Extend the left and right side.
        for i in range(0, times):
            # First, extend all edges.
            left_tile_arrs = self.extend_one_edge(left_tile_arrs, 'left', ratio)
            right_tile_arrs = self.extend_one_edge(right_tile_arrs, 'right', ratio)

            # Recompute the boxes
            tmp_box = shift_box(tile_boxes['right_tiles'], tile_size, 'right')
            tmp_box.pop()
            tmp_box.pop()
            tile_boxes = {'left_tiles': tile_boxes['left_tiles'],
                          'right_tiles': tmp_box}

            # Merge into a new image.
            prev_height, prev_width = prev_img.size
            prev_img_box = (tile_size, 0, tile_size + prev_height, prev_width)
            new_img = Image.new(mode='RGB', size=(prev_height + 2 * tile_size, prev_width))
            paste_img(new_img, left_tile_arrs, tile_boxes['left_tiles'])
            paste_img(new_img, right_tile_arrs, tile_boxes['right_tiles'])
            new_img.paste(prev_img, prev_img_box)

            prev_img = new_img

            print("Iteration " + str(i) + " Complete.")

        return new_img

    def extend_all_v3(self, meta_img_path, tile_size, ratio, times):
        raw_img = img_open(meta_img_path)
        width, height = raw_img.size

        #
        prev_img = raw_img
        new_img = prev_img

        # Prepare the images.
        tile_boxes = divide_image(height, width, tile_size)
        up_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['up_tiles']))
        down_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['down_tiles']))
        left_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['left_tiles']))
        right_tile_arrs = imgs2arr(cut_image(raw_img, tile_boxes['right_tiles']))

        # Extend.
        for i in range(0, times):
            # Extend upper tiles.
            up_first_tile = self.extend_corner(up_tile_arrs[0], 'up', ratio)


