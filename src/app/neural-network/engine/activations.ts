import {IActivation, ISingleValueActivation} from "./base";
import {Matrix1D} from "./matrix";
import {Param} from "../serialization";
import * as Matrix from "./matrix";
import * as Iter from "./iter";

export abstract class ActivationCombinedBase implements IActivation, ISingleValueActivation {
    abstract value(x: number): number;
    abstract moment(x: number): number;

    forward(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return Matrix.matrix1d_unary_op(input, this.value.bind(this), dst);
    }

    backward(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return Matrix.matrix1d_unary_op(input, this.moment.bind(this), dst);
    }
}

export class SigmoidActivation extends ActivationCombinedBase {
    value(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    moment(x: number): number {
        const s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
    }
}

export class LeakyReluActivation extends ActivationCombinedBase {
    @Param()
    alpha: number;

    constructor({alpha = 0.3} = {}) {
        super();
        this.alpha = alpha
    }

    value(x: number): number {
        return x >= 0 ? x : x * this.alpha;
    }

    moment(x: number): number {
        return x >= 0 ? 1 : this.alpha;
    }
}

export class ReluActivation extends ActivationCombinedBase {
    value(x: number): number {
        return x >= 0 ? x : 0;
    }

    moment(x: number): number {
        return x >= 0 ? 1 : 0;
    }
}

export class TanhActivation extends ActivationCombinedBase {
    value(x: number): number {
        return Math.tanh(x);
    }

    moment(x: number): number {
        return 1 - Math.pow(Math.tanh(x), 2);
    }
}

export class LinearActivation extends ActivationCombinedBase {
    value(x: number): number {
        return x
    }

    moment(_: number): number {
        return 1;
    }
}

export class SoftMaxActivation implements IActivation {
    forward(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        const max = Iter.max(input);
        const mapped = Matrix.matrix1d_unary_op(input, value => Math.exp(value - max), dst);
        const sum = Iter.sum(mapped);

        return Matrix.matrix1d_unary_in_place_op(mapped, v => v / sum);
    }

    backward(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return Matrix.matrix1d_unary_op(input, () => 1, dst);
    }
}

export type ActivationT = "sigmoid" | "relu" | "leakyRelu" | "tanh" | "linear" | "softmax";

export const ActivationsMap = {
    sigmoid: SigmoidActivation,
    relu: ReluActivation,
    leakyRelu: LeakyReluActivation,
    tanh: TanhActivation,
    linear: LinearActivation,
    softmax: SoftMaxActivation,
};

export const Activations = {
    SigmoidActivation,
    ReluActivation,
    LeakyReluActivation,
    TanhActivation,
    LinearActivation,
    SoftMaxActivation,
}