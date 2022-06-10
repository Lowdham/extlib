import torch


class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class tile:
    def __init__(self, ul_pnt, dr_pnt):
        self.ul_pnt = ul_pnt
        self.dr_pnt = dr_pnt

    def get_box(self):
        return self.ul_pnt.x, self.ul_pnt.y, self.dr_pnt.x, self.dr_pnt.y