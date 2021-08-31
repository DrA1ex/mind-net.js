import * as matrix from "./engine/matrix";

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
        const accuracy = mse(t_output, out)

        console.log(`INPUT ${t_input} OUTPUT ${outValue} EXPECTED ${tOutValue} (accuracy ${(accuracy * 100).toFixed(2)}%)`);

        accuracySum += accuracy;
    }

    console.log(`TOTAL ACCURACY: ${(accuracySum / training_data.length).toFixed(2)}%`);
    console.log('***');
}
