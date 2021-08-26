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
        return matrix.matrix1d_binary_op(activations, error, (a, e) => layer.activation.moment(a) * e * this.lr);
    }
}

type NesterovCacheT = { moments: matrix.Matrix1D, tmp1: matrix.Matrix1D };

class SgdNesterovOptimizer implements IOptimizer {
    readonly description: string;
    readonly lr: number;
    readonly beta: number;

    private cache = new Map<ILayer, NesterovCacheT>();

    constructor(beta = 0.9, lr = 0.01) {
        this.lr = lr;
        this.beta = beta;

        this.description = `nesterov(beta: ${this.beta}, lr: ${this.lr})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                moments: matrix.zero(layer.size),
                tmp1: matrix.zero(layer.size)
            })
        }

        const s = this.cache.get(layer)!;

        // next gradient
        matrix.matrix1d_binary_op(activations, s.moments, (a, m) => layer.activation.moment(a + m), s.tmp1);
        matrix.mul_to(s.tmp1, error);

        matrix.matrix1d_binary_in_place_op(s.moments, s.tmp1, (m, g) => this.beta * m + this.lr * g);

        return s.moments;
    }
}


type RMSPropCacheT = { velocities: matrix.Matrix1D, tmp1: matrix.Matrix1D };

class RMSPropOptimizer implements IOptimizer {
    readonly description: string;
    readonly beta: number;
    readonly lr: number;
    readonly eps: number

    private cache = new Map<ILayer, RMSPropCacheT>();

    constructor(beta = 0.9, lr = 0.005, eps = 1e-8) {
        this.beta = beta;
        this.lr = lr;
        this.eps = eps;
        this.description = `rmsprop(beta: ${beta}, lr: ${this.lr}, eps: ${this.eps})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                velocities: matrix.zero(layer.size),
                tmp1: matrix.zero(layer.size)
            })
        }

        const s = this.cache.get(layer)!;

        // next gradient
        matrix.matrix1d_binary_op(activations, error, (a, e) => e * layer.activation.moment(a), s.tmp1);

        matrix.matrix1d_binary_in_place_op(s.velocities, s.tmp1, (m, g) => this.beta * m + (1 - this.beta) * g * g);
        matrix.matrix1d_binary_in_place_op(s.tmp1, s.velocities, (g, m) => (this.lr / Math.sqrt(m + this.eps) * g));

        return s.tmp1;
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

        // gradient
        matrix.matrix1d_binary_op(activations, error, (a, e) => layer.activation.moment(a) * e, s.tmp1);

        // moving smooth moment and velocity
        matrix.matrix1d_binary_in_place_op(s.moments, s.tmp1, (m, g) => m * this.beta1 + (1 - this.beta1) * g);
        matrix.matrix1d_binary_in_place_op(s.velocities, s.tmp1, (m, g) => m * this.beta2 + (1 - this.beta2) * g * g);

        // boost moment and velocity for first epochs
        matrix.matrix1d_binary_in_place_op(s.tmp1, s.moments, (_, m) => m / (1 - Math.pow(this.beta1, epoch + 1)));
        matrix.matrix1d_binary_in_place_op(s.tmp2, s.velocities, (_, v) => v / (1 - Math.pow(this.beta2, epoch + 1)));

        matrix.matrix1d_binary_in_place_op(s.tmp1, s.tmp2, (m, v) => (m * this.lr) / (Math.sqrt(v) + this.eps));

        return s.tmp1;
    }

}

export type OptimizerT = "sgd" | "nesterov" | "adam" | "rmsprop";

export const Optimizers = {
    sgd: SgdOptimizer,
    nesterov: SgdNesterovOptimizer,
    adam: AdamOptimizer,
    rmsprop: RMSPropOptimizer
}
