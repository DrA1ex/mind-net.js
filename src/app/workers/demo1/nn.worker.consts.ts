import * as color from "../../utils/color"

export interface Point {
    type: number;
    x: number;
    y: number
}

export const DRAWING_DELAY = 1000 / 24;
export const MAX_ITERATION_TIME = DRAWING_DELAY / 4;

export const TRAINING_BATCH_SIZE = 10000;
export const MAX_TRAINING_ITERATION = 5e6;

export const DESIRED_RESOLUTION_X = 640;
export const DESIRED_RESOLUTION_Y = 480;
export const RESOLUTION_SCALE = 1 / 4;


export const DEFAULT_NN_LAYERS = [13, 7, 5];
export const DEFAULT_LEARNING_RATE = 0.01;


export const COLOR_A_HEX = "#e72525";
export const COLOR_A_BIN = color.getBinFromHex(COLOR_A_HEX);
export const COLOR_B_HEX = "#2562e7";
export const COLOR_B_BIN = color.getBinFromHex(COLOR_B_HEX);
