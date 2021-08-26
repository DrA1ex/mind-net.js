import * as matrix from "./matrix";

import {ILayer, IOptimizer} from "./base";

class SgdOptimizer implements IOptimizer {
    readonly description: string;
    readonly lr: number;

    constructor(lr = 0.01) {
        this.lr = lr;
        this.description = `sgd(lr: ${this.lr})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        const dst = layer.activation.moment(activations);
        matrix.mul_to(dst, error);
        matrix.matrix1d_unary_in_place_op(dst, g => g * this.lr);

        return dst;
    }
}

class SgdNesterovOptimizer implements IOptimizer {
    readonly description: string;
    readonly lr: number;
    readonly beta: number;

    private cache = new Map<ILayer, matrix.Matrix1D>();

    constructor(beta = 0.9, lr = 0.01) {
        this.lr = lr;
        this.beta = beta;

        this.description = `nesterov(beta: ${this.beta}, lr: ${this.lr})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, matrix.zero(layer.size))
        }

        const moment = this.cache.get(layer)!;

        let dst = matrix.add(activations, moment);
        layer.activation.moment(dst, dst)
        matrix.mul_to(dst, error);

        matrix.matrix1d_binary_in_place_op(moment, dst, (m, g) => this.beta * m + this.lr * g);

        return moment;
    }
}

type AdamCacheT = { moments: matrix.Matrix1D, velocities: matrix.Matrix1D, tmp1: matrix.Matrix1D, tmp2: matrix.Matrix1D };

class AdamOptimizer implements IOptimizer {
    readonly description: string;
    readonly lr: number;
    readonly beta1: number;
    readonly beta2: number;
    readonly eps: number;

    private cache = new Map<ILayer, AdamCacheT>();

    constructor(beta1 = 0.9, beta2 = 0.999, lr = 0.005, eps = 1e-8) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;

        this.description = `adam(beta1: ${beta1}, beta2: ${beta2}, lr: ${this.lr}, eps: ${eps.toExponential()})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer) || epoch === 0) {
            this.cache.set(layer, {
                moments: matrix.zero(layer.size),
                velocities: matrix.zero(layer.size),
                tmp1: matrix.zero(layer.size),
                tmp2: matrix.zero(layer.size)
            })
        }

        const s = this.cache.get(layer)!;

        layer.activation.moment(activations, s.tmp1);
        const gradient = matrix.mul_to(s.tmp1, error);

        matrix.matrix1d_binary_in_place_op(s.moments, gradient, (m, g) => m * this.beta1 + (1 - this.beta1) * g);
        matrix.matrix1d_binary_in_place_op(s.velocities, gradient, (m, g) => m * this.beta2 + (1 - this.beta2) * g * g);

        matrix.matrix1d_binary_in_place_op(s.tmp1, s.moments, (_, m) => m / (1 - Math.pow(this.beta1, epoch + 1)));
        matrix.matrix1d_binary_in_place_op(s.tmp2, s.velocities, (_, v) => v / (1 - Math.pow(this.beta2, epoch + 1)));

        matrix.matrix1d_binary_in_place_op(s.tmp1, s.tmp2, (m, v) => (m * this.lr) / (Math.sqrt(v) + this.eps));

        return s.tmp1;
    }

}

export type OptimizerT = "sgd" | "nesterov" | "adam";

export const Optimizers = {
    sgd: SgdOptimizer,
    nesterov: SgdNesterovOptimizer,
    adam: AdamOptimizer
}
