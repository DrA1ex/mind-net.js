import {IActivation} from "./base";
import {Matrix1D} from "./matrix";
import * as matrix from "./matrix";
import * as iter from "./iter";

export class SigmoidActivation implements IActivation {
    value(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_unary_op(input, x => 1 / (1 + Math.exp(-x)), dst);
    }

    moment(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_unary_in_place_op(this.value(input, dst), s => s * (1 - s));
    }
}

export class LeakyReluActivation implements IActivation {
    alpha: number;

    constructor(alpha = 0.3) {
        this.alpha = alpha
    }

    value(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_unary_op(input, x => x > 0 ? x : x * this.alpha, dst);
    }

    moment(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_unary_op(input, x => x > 0 ? 1 : this.alpha, dst);
    }
}

export class ReluActivation implements IActivation {
    private leakyRelu = new LeakyReluActivation(0);

    value(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return this.leakyRelu.value(input, dst);
    }

    moment(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return this.leakyRelu.moment(input, dst);
    }
}

export class TanhActivation implements IActivation {
    value(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_unary_op(input, Math.tanh, dst);
    }

    moment(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_unary_op(input, x => 1 - Math.pow(Math.tanh(x), 2), dst);
    }
}

export class LinearActivation implements IActivation {
    value(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return dst ? matrix.copy_to(input, dst) : input;
    }

    moment(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return dst ? matrix.copy_to(input, dst) : input;
    }
}

export class SoftMaxActivation implements IActivation {
    value(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        const max = iter.max(input);
        const mapped = matrix.matrix1d_unary_op(input, value => Math.exp(value - max), dst);
        const sum = iter.sum(mapped);

        return matrix.matrix1d_unary_in_place_op(mapped, v => v / sum);
    }

    moment(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return matrix.matrix1d_unary_op(input, () => 1, dst);
    }
}

export type ActivationT = "sigmoid" | "relu" | "leakyRelu" | "tanh" | "linear" | "softmax";

export const Activations = {
    sigmoid: SigmoidActivation,
    relu: ReluActivation,
    leakyRelu: LeakyReluActivation,
    tanh: TanhActivation,
    linear: LinearActivation,
    softmax: SoftMaxActivation,
};