import {ILoss} from "./base";
import {Matrix1D} from "./matrix";
import * as matrix from "./matrix";
import * as iter from "./iter";
import * as utils from "../utils";

export class MeanSquaredErrorLoss implements ILoss {
    loss(predicted: Matrix1D[], expected: Matrix1D[]): number {
        const rows = predicted.length;
        const columns = predicted[0].length;

        let sum = 0;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < columns; j++) {
                sum += Math.pow(expected[i][j] - predicted[i][j], 2) / rows / columns;
            }
        }

        return sum;
    }

    accuracy(predicted: Matrix1D[], expected: Matrix1D[]): number {
        return utils.absoluteAccuracy(predicted, expected);
    }

    calculateError(predicted: Matrix1D, expected: Matrix1D): Matrix1D {
        //TODO: cache
        return matrix.matrix1d_binary_op(predicted, expected, (p, e) =>
            -2 * (e - p) / predicted.length);
    }
}

export class MeanAbsoluteErrorLoss implements ILoss {
    loss(predicted: Matrix1D[], expected: Matrix1D[]): number {
        const rows = predicted.length;
        const columns = predicted[0].length;

        let sum = 0;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < columns; j++) {
                sum += (expected[i][j] - predicted[i][j]) / rows / columns;
            }
        }

        return sum;
    }

    accuracy(predicted: Matrix1D[], expected: Matrix1D[]): number {
        return utils.absoluteAccuracy(predicted, expected);
    }

    calculateError(predicted: Matrix1D, expected: Matrix1D): Matrix1D {
        //TODO: cache
        return matrix.matrix1d_binary_op(predicted, expected, (p, e) =>
            Math.sign(p - e) / predicted.length);
    }
}

export class CategoricalCrossEntropyLoss implements ILoss {
    loss(predicted: Matrix1D[], expected: Matrix1D[]): number {
        const rows = predicted.length;
        const columns = predicted[0].length;

        const softMaxed = predicted.map(this._softMax);

        let sum = 0;
        for (let i = 0; i < rows; i++) {
            let rowSum = 0;
            for (let j = 0; j < columns; j++) {
                rowSum += this._clip(softMaxed[i][j]) * expected[i][j];
            }

            sum += -Math.log(rowSum);
        }

        return sum / rows;
    }

    accuracy(predicted: Matrix1D[], expected: Matrix1D[]): number {
        const rows = predicted.length;

        let count = 0;
        for (let i = 0; i < rows; i++) {
            const pMaxIndex = predicted[i].indexOf(Math.max(...predicted[i]));
            const eMaxIndex = expected[i].indexOf(Math.max(...expected[i]));

            if (pMaxIndex === eMaxIndex) {
                count += 1;
            }
        }

        return count / rows;
    }

    calculateError(predicted: Matrix1D, expected: Matrix1D): Matrix1D {
        //TODO: cache
        const softMaxed = this._softMax(predicted);
        return matrix.sub(softMaxed, expected);
    }

    private _clip(value: number, eps = 1e-7) {
        return Math.max(eps, Math.min(value, 1 - eps))
    }

    private _softMax(data: Matrix1D): Matrix1D {
        const max = iter.max(data);
        const mapped = matrix.matrix1d_unary_op(data, value => Math.exp(value - max));
        const sum = iter.sum(mapped);

        return matrix.matrix1d_unary_in_place_op(mapped, v => v / sum);
    }
}

export function buildLoss(loss: LossT | ILoss = 'mse') {
    const loss_param = typeof loss === "string" ? Loss[loss] : loss
    if (!loss_param) {
        throw new Error(`Unknown loss type ${loss_param}`);
    }

    if (typeof loss_param === "object") {
        return loss_param;
    }

    return new loss_param();
}

export type LossT = "mse" | "mae" | "categoricalCrossEntropy";
export const Loss = {
    mse: MeanSquaredErrorLoss,
    mae: MeanAbsoluteErrorLoss,
    categoricalCrossEntropy: CategoricalCrossEntropyLoss
}