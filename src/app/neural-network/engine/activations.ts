import * as matrix from "./matrix";
import {OptMatrix1D} from "./matrix";

import {IActivation} from "./base";

class SigmoidActivation implements IActivation {
    value(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        return matrix.matrix1d_unary_op(m, x => 1 / (1 + Math.exp(-x)), dst);
    }

    moment(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        const sig = this.value(m, dst);
        return matrix.matrix1d_unary_op(sig, x => x * (1 - x), sig);
    }
}

class LeakyReluActivation implements IActivation {
    alpha: number;

    constructor(alpha = 0.1) {
        this.alpha = alpha
    }

    value(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        return matrix.matrix1d_unary_op(m, x => x > 0 ? x : x * this.alpha, dst);
    }

    moment(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        return matrix.matrix1d_unary_op(m, x => x > 0 ? 1 : this.alpha, dst);
    }
}

class ReluActivation implements IActivation {
    private leakyRelu = new LeakyReluActivation(0);

    value(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        return this.leakyRelu.value(m, dst);
    }

    moment(m: matrix.Matrix1D, dst: OptMatrix1D = undefined): matrix.Matrix1D {
        return this.leakyRelu.moment(m, dst);
    }
}

export type ActivationT = "sigmoid" | "relu" | "leakyRelu";

export const Activations = {
    sigmoid: SigmoidActivation,
    relu: ReluActivation,
    leakyRelu: LeakyReluActivation
};