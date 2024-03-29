import * as color from "../../utils/color";

export type NetworkParams = [number, number[], number];

export const DEFAULT_LEARNING_RATE = 0.01;
export const DEFAULT_NN_PARAMS: NetworkParams = [32, [64, 128], 16 * 16];
export const DEFAULT_BATCH_SIZE = 64;

export const DRAWING_DELAY = 1000;
export const MAX_ITERATION_TIME = DRAWING_DELAY / 2;
export const PROGRESS_DELAY = DRAWING_DELAY / 3;

export const DRAW_GRID_DIMENSION = 10;

export const COLOR_A_BIN = color.getBinFromHex("#000000");
export const COLOR_B_BIN = color.getBinFromHex("#ffffff");
