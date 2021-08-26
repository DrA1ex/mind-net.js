import * as iter from "../utils/iter";
import * as matrix from "../utils/matrix";
// @ts-ignore
import * as nn from "./sequential";

export function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

export function der_sigmoid(v: matrix.Matrix1D): matrix.Matrix1D {
    return matrix.matrix1d_unary_op(v, a => a * (1 - a));
}

export function leakyReLU(x: number, alpha: number = 0.01) {
    return x > 0 ? x : x * alpha;
}

export function der_leakyReLU(v: matrix.Matrix1D, alpha = 0.01) {
    return matrix.matrix1d_unary_op(v, x => x >= 0 ? 1 : alpha);
}

export function print(nn: any, training_data: [matrix.Matrix1D, matrix.Matrix1D][]) {
    if (training_data.length === 0) {
        return;
    }

    console.log('***');

    let accuracySum = 0;
    for (let i = 0; i < training_data.length; i++) {
        const [t_input, t_output] = training_data[i];
        const out: any[] = nn.compute(t_input);

        const outValue = out.map(n => n.toFixed(2));
        const tOutValue = t_output.map(n => n.toFixed(2));
        const accuracy = iter.zip(t_output, out).map(([a, b]) => (100 - Math.abs(a - b) * 100))
        const accuracyValue = accuracy.map(v => v.toFixed(1));

        console.log(`INPUT ${t_input} OUTPUT ${outValue} EXPECTED ${tOutValue} (accuracy ${accuracyValue}%)`);

        accuracySum += matrix.sum(accuracy);
    }

    console.log(`TOTAL ACCURACY: ${(accuracySum / training_data.length / training_data[0][1].length).toFixed(2)}%`);
    console.log('***');
}
