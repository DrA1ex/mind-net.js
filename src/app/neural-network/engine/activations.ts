import {IActivation} from "./base";

class SigmoidActivation implements IActivation {
    value(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    moment(x: number): number {
        const s = this.value(x);
        return s * (1 - s);
    }
}

class LeakyReluActivation implements IActivation {
    alpha: number;

    constructor(alpha = 0.01) {
        this.alpha = alpha
    }

    value(x: number): number {
        return x > 0 ? x : x * this.alpha;
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

    moment(x: number): number {
        return this.leakyRelu.moment(x);
    }
}

class TanhActivation implements IActivation {
    value(x: number): number {
        return Math.tanh(x);
    }

    moment(x: number): number {
        return 1 - Math.pow(Math.tanh(x), 2);
    }
}

export type ActivationT = "sigmoid" | "relu" | "leakyRelu" | "tanh";

export const Activations = {
    sigmoid: SigmoidActivation,
    relu: ReluActivation,
    leakyRelu: LeakyReluActivation,
    tanh: TanhActivation
};