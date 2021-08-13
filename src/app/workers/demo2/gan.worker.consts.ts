import * as color from "../../utils/color";

export type NetworkParams = [number, number[], number, number[]];

export const DRAWING_DELAY = 1000;
export const MAX_ITERATION_TIME = DRAWING_DELAY / 2;
export const TRAINING_BATCH_SIZE = 500;

export const COLOR_A_BIN = color.getBinFromHex("#000000");
export const COLOR_B_BIN = color.getBinFromHex("#ffffff");
