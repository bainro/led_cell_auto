#!/usr/bin/env python
# -*- coding: utf-8 -*-
from samplebase import SampleBase
from rgbmatrix import graphics
from random import randint
import numpy as np
import cv2 as cv
from scipy import signal

class Automata(SampleBase):

    rule_num   = 110 	    # controls which cellular automate is generated (E.g. rule 110)
    dir_toggle = False	    # direction control toggle, <- or ->
    sleep_time = 5e4        # larger numbers make the scrolling slower
    rand_col_1 = False      # whether the first col in random or just the top point
    color_mode = True       # Toggles type of pixel coloring
    img_bckgnd = True       # Use image as background. Currently rainbow.png
    img_c_mode = True 	    # Switches between image channel formats e.g. RGB & BRG
    img_flip_y = True	    # Flips image across the y-axis.
    img_flip_x = False	    # Flips image across the y-axis.
    img_only   = True      # Just draws the image, no cellular automata
    img        = "imgs/tool.png"

    colors = {
        "red":          [255,   0,   0],
        "black":        [  0,   0,   0],
        "white":        [255, 255, 255],
        "yellow":       [255, 255,   0],
        "green":        [  0, 255,   0],
        "gray_1":       [ 55,  55,  55],
        "gray_2":       [ 25,  25,  25],
        "gray_3":       [ 15,  15,  15],
        "green_1":      [  0,  30,   0],
        "green_2":      [ 17,  65,  18],
        "green_3":      [ 20,  55,   4],
        "blue_1":       [ 20,   7,  70],
        "blue_2":       [ 12,   5,  40],
        "blue_3":       [  7,   4,  36],
        "red_1":        [130,   0,   0],
        "red_2":        [ 50,   2,   0],
        "red_3":        [ 35,   1,   0],
        "red_4":        [ 25,   0,   0],
    }

    # color sets to choose from
    blue_colors = ["blue_1", "blue_2", "blue_3"]
    green_colors = ["green_1", "green_2", "green_3"]
    red_colors = ["red_1", "red_2", "red_3"]
    gray_colors = ["gray_1", "gray_2", "gray_3"]
    OMG_colors = ["red", "green", "yellow"]
    test_colors = ["red_2", "red_3", "red_4"]
    color_set = green_colors

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
        if self.rand_col_1:
            self.board[-1,:,0] = [randint(0,1) for _ in range(self.board.shape[1])]
        else:
            self.board[-1,-1] = 1
        self.col_neighbors = np.array([1, 2, 4], dtype=np.uint8)
        # format a decimal number as binary then reverse it using the [] op.
        rule = "{0:08b}".format(self.rule_num)[::-1]
        self.rule_kernel = np.array([int(x) for x in rule], dtype=np.uint8)

        if self.color_mode:
            if self.img_bckgnd:
                img = cv.imread(self.img)
                img = cv.resize(img, (128, 32))
                # some images require swapping color channels. e.g. BRG->RGB, etc
                img = np.rot90(img//1) # division lowers brightness
                self.board[:,:,1:] = img

                if self.img_flip_y:
                    self.board[:,:,1:] = np.flip(self.board[:,:,1:],0)
                if self.img_flip_x:
                    self.board[:,:,1:] = np.flip(self.board[:,:,1:],1)

                tmp = np.array(self.board[:,:,1:], dtype=np.uint8)
                if self.img_c_mode:
                        self.board[:,:,1] = tmp[:,:,2]
                        self.board[:,:,2] = tmp[:,:,1]
                        self.board[:,:,3] = tmp[:,:,0]
            else:
                # set each position's/pixel's color.
                for col_i in range(self.board.shape[0]):
                    for row_i in range(self.board.shape[1]):
                        rand_color_i = randint(0, len(self.color_set)-1)
                        self.board[col_i, row_i, 1:] = self.color_set[rand_color_i]

    def step(self):
        if self.img_only:
            self.board[:,:,0] = 1
            return

        # create bottom col
        cur_col = np.reshape(self.board[-1,:,0], self.board.shape[1])

        # shift & cut off top col
        if self.color_mode:
            self.board[:-1,:,0] = self.board[1:,:,0]
        else:
            self.board[:-1,:,:] = self.board[1:,:,:]

        rule_index = signal.convolve2d(
                         cur_col[None,:],
                         self.col_neighbors[None,:],
                         mode='same',
                         boundary='wrap')
        next_col = self.rule_kernel[rule_index[0]]
        self.board[-1,:,0] = next_col

        """
        # inject some random deaths to keep 110 from looping
        if self.rule_num == 110 and randint(0,300) == 69:
            #rand_col_i = randint(0, self.board.shape[0]-1)
            #self.board[rand_col_i,:,0] = 0
            # black out every even col
            #self.board[::2,::2,0] = 0
            rand_row_i = randint(0, self.board.shape[1]-1)
            self.board[-1,:,0] = 0
            self.board[-1,rand_row_i,0] = 1
        """

        # This way causes the pixels to change color at each step
        # assign color values to the new col
        if not self.color_mode:
            for _row_i in range(self.board.shape[1]):
                rand_color_i = randint(0, len(self.color_set)-1)
                self.board[-1, :, 1:] = self.color_set[rand_color_i]

    def render(self):
        r_, g_, b_ = 0, 0, 0

        # Allows us to easily break out of the program with Ctrl-c
        try:
            while True:
                self.offset_canvas.Clear()
                self.step()

                if self.sleep_time > 0:
                    self.usleep(self.sleep_time) # easy way to tune FPS

                for col_i in range(self.board.shape[0]):
                    for row_i in range(self.board.shape[1]):
                        if self.board[col_i, row_i, 0] >= 1:
                            r_, g_, b_ = self.board[col_i, row_i, 1:4]
                            if self.dir_toggle:
                                _col_i = abs(col_i - self.board.shape[0] + 1)
                                _row_i = abs(row_i - self.board.shape[1] + 1)
                                self.offset_canvas.SetPixel(_col_i, _row_i, r_, g_, b_)
                            else:
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
