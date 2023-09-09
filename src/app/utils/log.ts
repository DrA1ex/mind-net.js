import {ModelBase} from "../neural-network/engine/models/base";
import * as matrix from "../neural-network/engine/matrix";
import {Iter} from "../neural-network/neural-network";
import {Matrix1D} from "../neural-network/engine/matrix";

export function print(nn: ModelBase, input: matrix.Matrix1D[], expected: matrix.Matrix1D[]) {
    if (input.length === 0) {
        return;
    }

    console.log('***');

    let accuracySum = 0;
    for (let i = 0; i < input.length; i++) {
        const t_input = input[i];
        const t_output = expected[i];
        const out: Matrix1D = nn.compute(t_input);

        const outValue = Array.from(Iter.map(out, n => n.toFixed(2)));
        const tOutValue = Array.from(Iter.map(t_output, n => n.toFixed(2)));
        const accuracy = nn.loss.accuracy([out], [t_output]) * 100;

        console.log(`INPUT ${t_input} OUTPUT ${outValue} EXPECTED ${tOutValue} (accuracy ${accuracy.toFixed(2)}%)`);

        accuracySum += accuracy;
    }

    console.log(`TOTAL ACCURACY: ${(accuracySum / input.length).toFixed(2)}%`);
    console.log('***');
}
