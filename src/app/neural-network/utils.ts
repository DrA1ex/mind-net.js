import * as matrix from "./engine/matrix";
import {ModelBase} from "./engine/models/base";

export function mse(actual: matrix.Matrix1D, expected: matrix.Matrix1D) {
    let sum = 0
    for (let i = 0; i < actual.length; i++) {
        sum += Math.pow(expected[i] - actual[i], 2);
    }

    return sum / actual.length;
}

export function generateInputNoise(size: number, from = 0, to = 1): matrix.Matrix1D {
    return matrix.random(size, from, to);
}

export function pickRandomItem(trainingData: matrix.Matrix1D[]) {
    return trainingData[Math.floor(Math.random() * trainingData.length)]
}

export function loss(nn: ModelBase, input: matrix.Matrix1D[], expected: matrix.Matrix1D[]): number {
    let sum = 0;
    for (let i = 0; i < input.length; i++) {
        const actual = nn.compute(input[i]);
        sum += mse(actual, expected[i]);
    }

    return sum / input.length;
}
