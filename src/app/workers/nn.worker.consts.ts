export interface Point {
    type: number;
    x: number;
    y: number
}

export const ITERATIONS_PER_CALL = 80000;
export const MAX_TRAINING_ITERATION = ITERATIONS_PER_CALL * 10000;

export const X_STEP = 1 / 100;
export const Y_STEP = 1 / 100;
