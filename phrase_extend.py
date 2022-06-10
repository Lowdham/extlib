from extlib.preprocess import *


class test_gen:
    def __init__(self):
        self.color = 0
        self.layer = 0
        self.cnt = 0

    def __call__(self, arg):
        s = arg.shape[0]
        c = arg.shape[1]
        h = arg.shape[2]
        w = arg.shape[3]
        arr = np.zeros((s, 3, h, w))
        arr[:, self.layer, :, :] = 12346
        self.cnt = self.cnt + 1
        if self.cnt == 1024:
            self.color = self.color + 1
            self.cnt = 0
            if self.layer == 2:
                self.layer = 0
            else:
                self.layer = self.layer + 1
        return torch.from_numpy(arr).to("cpu")


#
class extend_engine:
    def __init__(self, generator, device):
        self.generator = generator
        self.device = device

    def extend_once(self, image_tensor, img, mask):
        with torch.no_grad():
            gen = self.generator(image_tensor)
            result = gen * mask + img * (1 - mask)
            return (result + 1) * 0.5

    def extend_one_edge(self, tiles, mode, ratio):
        # Convert tiles into the numpy array.
        old_tiles = normalize_images(tiles)
        new_tiles = []

        for i in range(0, int(1 / ratio)):
            for unmask_tile in old_tiles:
                img_tensor, img, mask = mask_tile(unmask_tile, mode, ratio, self.device)
                generated_img = self.extend_once(img_tensor, img, mask).squeeze()
                new_tiles.append(generated_img.cpu().numpy())

            old_tiles = new_tiles
            new_tiles = []

        return unnormalize_images(old_tiles)

    def extend_corner(self, base_tile, mode, ratio):
        result_tile = normalize_image(base_tile)
        for i in range(0, int(1 / ratio)):
            img_tensor, img, mask = mask_tile(result_tile, mode, ratio, self.device)
            result_tile = self.extend_once(img_tensor, img, mask).squeeze().cpu().numpy()

        return unnormalize_image(result_tile)

    def extend_all(self, meta_img_path, tile_size, ratio, times):
        raw_img = img_open(meta_img_path)
        height, width = raw_img.size

        # Check.

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
            prev_height, prev_width = prev_img.size
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

