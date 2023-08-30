import * as Matrix from "../engine/matrix";

export function pickRandomItem(trainingData: Matrix.Matrix1D[]) {
    return trainingData[Math.floor(Math.random() * trainingData.length)]
}

export function absoluteAccuracy(input: Matrix.Matrix1D[], expected: Matrix.Matrix1D[], k = 2.5) {
    if (!expected) return 0;

    let precision;
    if (expected[0].length > 1) {
        precision = std(expected.map(r => std(r))) / k;
    } else {
        precision = std(expected.flat()) / k;
    }

    const rows = input.length;
    const columns = input[0].length;

    let outSum = 0;
    for (let i = 0; i < rows; i++) {
        let sum = 0;
        for (let j = 0; j < columns; j++) {
            sum += Math.abs(input[i][j] - expected[i][j])
        }

        outSum += sum / columns < precision ? 1 : 0;
    }

    return outSum / rows;
}

export function std(data: Matrix.Matrix1D) {
    const length = data.length;
    const mean = data.reduce((p, c) => (p + c) / length, 0);
    const meanOfSquareDiff = data.reduce((p, c) => p + Math.pow(c - mean, 2) / length, 0);
    return Math.sqrt(meanOfSquareDiff);
}