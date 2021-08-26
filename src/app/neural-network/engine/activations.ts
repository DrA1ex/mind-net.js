import * as matrix from "./matrix";

import {IActivation} from "./base";

class SigmoidActivation implements IActivation {
    value(m: matrix.Matrix1D): matrix.Matrix1D {
        return matrix.matrix1d_unary_op(m, x => 1 / (1 + Math.exp(-x)));
    }

    moment(m: matrix.Matrix1D): matrix.Matrix1D {
        const sigm = this.value(m);
        return matrix.matrix1d_unary_op(sigm, x => x * (1 - x));
    }
}

class LeakyReluActivation implements IActivation {
    value(m: matrix.Matrix1D, alpha = 0.1): matrix.Matrix1D {
        return matrix.matrix1d_unary_op(m, x => x > 0 ? x : x * alpha);
    }

    moment(m: matrix.Matrix1D, alpha = 0.1): matrix.Matrix1D {
        return matrix.matrix1d_unary_op(m, x => x > 0 ? 1 : alpha);
    }
}

class ReluActivation implements IActivation {
    private leakyRelu = new LeakyReluActivation();

    value(m: matrix.Matrix1D): matrix.Matrix1D {
        return this.leakyRelu.value(m, 0);
    }

    moment(m: matrix.Matrix1D): matrix.Matrix1D {
        return this.leakyRelu.moment(m, 0);
    }
}

export type ActivationT = "sigmoid" | "relu" | "leakyRelu";

export const Activations = {
    sigmoid: SigmoidActivation,
    relu: ReluActivation,
    leakyRelu: LeakyReluActivation,
}