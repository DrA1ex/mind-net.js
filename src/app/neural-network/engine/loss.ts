import {ILoss} from "./base";
import {Matrix1D} from "./matrix";
import {Param} from "../serialization";
import * as matrix from "./matrix";
import * as utils from "../utils";

export class MeanSquaredErrorLoss implements ILoss {
    @Param()
    public k;

    constructor({k = 2.5} = {}) {
        this.k = k;
    }

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
        return utils.absoluteAccuracy(predicted, expected, this.k);
    }

    calculateError(predicted: Matrix1D, expected: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_binary_op(predicted, expected, (p, e) =>
            -2 * (e - p) / predicted.length, dst);
    }
}

export class MeanAbsoluteErrorLoss implements ILoss {
    @Param()
    public k;

    constructor({k = 2.5} = {}) {
        this.k = k;
    }

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
        return utils.absoluteAccuracy(predicted, expected, this.k);
    }

    calculateError(predicted: Matrix1D, expected: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_binary_op(predicted, expected, (p, e) =>
            Math.sign(e - p) / predicted.length, dst);
    }
}

export class CategoricalCrossEntropyLoss implements ILoss {
    loss(predicted: Matrix1D[], expected: Matrix1D[]): number {
        const rows = predicted.length;
        const columns = predicted[0].length;

        let sum = 0;
        for (let i = 0; i < rows; i++) {
            let rowSum = 0;
            for (let j = 0; j < columns; j++) {
                rowSum += this._clip(predicted[i][j]) * expected[i][j];
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

    calculateError(predicted: Matrix1D, expected: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.sub(predicted, expected, dst);
    }

    private _clip(value: number, eps = 1e-7) {
        return Math.max(eps, Math.min(value, 1 - eps))
    }
}

export class BinaryCrossEntropy implements ILoss {
    loss(predicted: Matrix1D[], expected: Matrix1D[]): number {
        const rows = predicted.length;
        const columns = predicted[0].length;

        let sum = 0;
        for (let i = 0; i < rows; i++) {
            let rowSum = 0;
            for (let j = 0; j < columns; j++) {
                const e = expected[i][j];
                const p = this._clip(predicted[i][j]);
                rowSum += -(e * Math.log(p) + (1 - e) * Math.log(1 - p));
            }

            sum += rowSum / columns;
        }

        return sum / rows;
    }

    accuracy(predicted: Matrix1D[], expected: Matrix1D[]): number {
        const rows = predicted.length;
        const columns = predicted[0].length;

        let count = 0;
        for (let i = 0; i < rows; i++) {
            let rowCount = 0;
            for (let j = 0; j < columns; j++) {
                const pState = predicted[i][j] > 0.5 ? 1 : 0;
                if (pState === expected[i][j]) {
                    rowCount += 1;
                }
            }

            count += rowCount / columns;
        }

        return count / rows;
    }

    calculateError(predicted: Matrix1D, expected: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_binary_op(predicted, expected, (p, e) => {
            const c = this._clip(p);
            return -(e / c - (1 - e) / (1 - c)) / predicted.length;
        }, dst);
    }

    private _clip(value: number, eps = 1e-7) {
        return Math.max(eps, Math.min(value, 1 - eps));
    }
}

export function buildLoss(loss: LossT | ILoss = 'mse') {
    const loss_param = typeof loss === "string" ? LossMap[loss] : loss
    if (!loss_param) {
        throw new Error(`Unknown loss type ${loss_param}`);
    }

    if (typeof loss_param === "object") {
        return loss_param;
    }

    return new loss_param();
}

export type LossT = "mse" | "mae" | "categoricalCrossEntropy" | "binaryCrossEntropy";
const LossMap = {
    mse: MeanSquaredErrorLoss,
    mae: MeanAbsoluteErrorLoss,
    categoricalCrossEntropy: CategoricalCrossEntropyLoss,
    binaryCrossEntropy: BinaryCrossEntropy
}

export const Loss = {
    MeanSquaredErrorLoss,
    MeanAbsoluteErrorLoss,
    CategoricalCrossEntropyLoss,
    BinaryCrossEntropy,
}