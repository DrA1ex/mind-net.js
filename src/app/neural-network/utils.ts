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

export function print(nn: ModelBase, input: matrix.Matrix1D[], expected: matrix.Matrix1D[]) {
    if (input.length === 0) {
        return;
    }

    console.log('***');

    let accuracySum = 0;
    for (let i = 0; i < input.length; i++) {
        const t_input = input[i];
        const t_output = expected[i];
        const out: any[] = nn.compute(t_input);

        const outValue = out.map(n => n.toFixed(2));
        const tOutValue = t_output.map(n => n.toFixed(2));
        const accuracy = (1 - mse(t_output, out)) * 100;

        console.log(`INPUT ${t_input} OUTPUT ${outValue} EXPECTED ${tOutValue} (accuracy ${accuracy.toFixed(2)}%)`);

        accuracySum += accuracy;
    }

    console.log(`TOTAL ACCURACY: ${(accuracySum / input.length).toFixed(2)}%`);
    console.log('***');
}
