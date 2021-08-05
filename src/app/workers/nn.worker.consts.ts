export interface Point {
    type: number;
    x: number;
    y: number
}

export const DRAWING_DELAY = 1000 / 24;
export const MAX_ITERATION_TIME = DRAWING_DELAY / 4;

export const MAX_TRAINING_ITERATION = 5e6;

export const X_STEP = 1 / 1280 * 8;
export const Y_STEP = 1 / 720 * 8;


export const DEFAULT_NN_LAYERS = [13, 7, 5];
export const DEFAULT_LEARNING_RATE = 0.01;