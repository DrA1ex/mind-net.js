export interface Point {
    type: number;
    x: number;
    y: number
}

export const DRAWING_DELAY = 1000 / 24;
export const MAX_ITERATION_TIME = DRAWING_DELAY / 4;

export const MAX_TRAINING_ITERATION = 5e6;

export const DESIRED_RESOLUTION_X = 640;
export const DESIRED_RESOLUTION_Y = 480;
export const RESOLUTION_SCALE = 1 / 4;


export const DEFAULT_NN_LAYERS = [13, 7, 5];
export const DEFAULT_LEARNING_RATE = 0.01;


export const COLOR_PATTERN_BIN = 0xff5800ce;
export const COLOR_PATTERN_RGB = "rgb(206,$value,88)";
