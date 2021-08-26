import * as matrix from "./matrix";
import {OptMatrix1D} from "./matrix";

import {IActivation} from "./base";

class SigmoidActivation implements IActivation {
    value(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    value_matrix(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        return matrix.matrix1d_unary_op(m, this.value, dst);
    }

    moment(x: number): number {
        const s = this.value(x);
        return s * (1 - s);
    }
}

class LeakyReluActivation implements IActivation {
    alpha: number;

    constructor(alpha = 0.1) {
        this.alpha = alpha
    }

    value(x: number): number {
        return x > 0 ? x : x * this.alpha;
    }

    value_matrix(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        return matrix.matrix1d_unary_op(m, (x) => this.value(x), dst);
    }

    moment(x: number): number {
        return x > 0 ? 1 : this.alpha;
    }
}

class ReluActivation implements IActivation {
    private leakyRelu = new LeakyReluActivation(0);

    value(x: number): number {
        return this.leakyRelu.value(x);
    }

    value_matrix(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        return this.leakyRelu.value_matrix(m, dst);
    }

    moment(x: number): number {
        return this.leakyRelu.moment(x);
    }
}

export type ActivationT = "sigmoid" | "relu" | "leakyRelu";

export const Activations = {
    sigmoid: SigmoidActivation,
    relu: ReluActivation,
    leakyRelu: LeakyReluActivation
};