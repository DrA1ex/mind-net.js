import * as iter from "./engine/iter";
import * as matrix from "./engine/matrix";

export function print(nn: any, training_data: [matrix.Matrix1D, matrix.Matrix1D][]) {
    if (training_data.length === 0) {
        return;
    }

    console.log('***');

    let accuracySum = 0;
    for (let i = 0; i < training_data.length; i++) {
        const [t_input, t_output] = training_data[i];
        const out: any[] = nn.compute(t_input);

        const t_output_array = Array.from(t_output.data);

        const outValue = out.map(n => n.toFixed(2));
        const tOutValue = t_output_array.map(n => n.toFixed(2));
        const accuracy = iter.zip(t_output_array, out).map(([a, b]) => (100 - Math.abs(a - b) * 100))
        const accuracyValue = accuracy.map(v => v.toFixed(1));

        console.log(`INPUT ${t_input} OUTPUT ${outValue} EXPECTED ${tOutValue} (accuracy ${accuracyValue}%)`);

        accuracySum += iter.sum(accuracy);
    }

    console.log(`TOTAL ACCURACY: ${(accuracySum / training_data.length / training_data[0][1].length).toFixed(2)}%`);
    console.log('***');
}
