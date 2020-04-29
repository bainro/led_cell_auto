#!/usr/bin/env python
# -*- coding: utf-8 -*-
from samplebase import SampleBase
from rgbmatrix import graphics
from random import randint
import numpy as np
from scipy import signal

class Automata(SampleBase):

    rule_num = 110          # controls which cellular automate is generated (E.g. rule 110)
    dir_toggle = True       # direction control toggle, <- or ->
    sleep_time = 5e4        # larger numbers make the scrolling slower

    colors = {
        "dark_red": [115, 18, 81],
        "red": [255, 0, 0],
        "black": [0, 0, 0],
        "white": [255, 255, 255],
        "gray": [55, 55, 55],
        "dark_blue": [20, 7, 70],
        "darker_blue": [12, 5, 40],
        "darkest_blue": [7, 4, 36],
        "green": [0, 255, 0],
        "camo_green": [34, 130, 37],
        "eww_green": [144, 255, 18],
    }

    # color sets to choose from
    blue_purple_colors = ["dark_blue", "darker_blue", "darkest_blue"]
    green_colors = ["green", "camo_green", "eww_green"]
    color_set = blue_purple_colors

    def __init__(self, gui = False, *args, **kwargs):
        super(Automata, self).__init__(*args, **kwargs)
        self.board = np.zeros((128,32,3), dtype=np.uint8)   # WHC format
        self.process()
        self.color_set = [self.colors[color_key] for color_key in self.color_set]
        self.color_set = np.array(self.color_set, dtype=np.uint8)


    # called by base class SampleBase to start the game
    def run(self):
        self.offset_canvas = self.matrix.CreateFrameCanvas()
        self.make_board()
        self.render()

    def make_board(self):
        self.board = np.zeros((128, 32, 4), dtype=np.uint8)
        # starts with just a single cell "alive", i.e. on
        self.board[127, 31] = 1
        self.col_neighbors = np.array([1, 2, 4], dtype=np.uint8)
        # format a decimal number as binary then reverse it using the [] op.
        rule = "{0:08b}".format(self.rule_num)[::-1]
        self.rule_kernel = np.array([int(x) for x in rule], dtype=np.uint8)

        # set each position's/pixel's color.
        for col_i in range(self.board.shape[0]):
            for row_i in range(self.board.shape[1]):
                rand_color_i = randint(0, len(self.color_set)-1)
                self.board[col_i, row_i, 1:] = self.color_set[rand_color_i]

    def step(self):
        # temp save col
        #cut_col = self.board[0,:,:]
        # shift & cut off top col
        self.board[:-1,:,:] = self.board[1:,:,:]
        # create bottom col
        cur_col = self.board[-2, :, 0]
        #cur_col = cut_col | self.board[126, :]
        #cur_col = np.bitwise_xor(cut_col, self.board[126, :])
        rule_index = signal.convolve2d(
                         cur_col[None, :],
                         self.col_neighbors[None, :],
                         mode='same',
                         boundary='wrap')
        next_col = self.rule_kernel[rule_index[0]]
        self.board[-1, :, 0] = next_col
        # This way causes the pixels to change color at each step
        # assign color values to the new col
        # for _row_i in range(self.board.shape[1]):
        #     rand_color_i = randint(0, len(self.colors)-1)
        #     self.board[-1, :, 1:] = self.colors[rand_color_i]

    def render(self):
        r_, g_, b_ = 0, 0, 0

        # Allows us to easily break out of the program with Ctrl-c
        try:
            while True:
                self.offset_canvas.Clear()

                self.usleep(self.sleep_time) # easy way to tune FPS
                self.step()

                for col_i in range(self.board.shape[0]):
                    for row_i in range(self.board.shape[1]):
                        if (self.board[col_i, row_i, 0] >= 1):
                            r_, g_, b_ = self.board[col_i, row_i, 1:4]
                            # (0, 31) is the (x, y) coordinates of the bottom left pixel when horizontal (128x32).
                            col_i = abs(col_i - self.board.shape[0] - 1)
                            self.offset_canvas.SetPixel(col_i, row_i, r_, g_, b_)

                self.offset_canvas = self.matrix.SwapOnVSync(self.offset_canvas)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    cell = Automata()
    cell.run()

"""
# Creates static board with all 128 cols used
for col_i in range(self.board.shape[0] - 1):
    cur_col = self.board[col_i, :]
    rule_index = signal.convolve2d(
                        cur_col[None, :],
                        col_neighbors[None, :],
                        mode='same',
                        boundary='wrap')
    next_col = rule_kernel[rule_index[0]]
    self.board[col_i + 1, :] = next_col

#self.font = graphics.Font()
#self.font.LoadFont("/home/pi/led_python/fonts/4x6.bdf")
#self.color_text = graphics.Color(150,150,150)

graphics.DrawText(self.offset_canvas, self.font, 102, 23, self.color_text, 'High')
graphics.DrawLine(self.offset_canvas, 0, 0, 0, 31, self.color_border)
self.offset_canvas.SetPixel(col_i, row_i, *self.colors.get("camo-green"))
"""
