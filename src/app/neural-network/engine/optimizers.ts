import * as matrix from "./matrix";

import {ILayer, IOptimizer} from "./base";

abstract class OptimizerBase implements IOptimizer {
    abstract readonly description: string;
    abstract readonly lr: number;

    abstract step(layer: ILayer, activations: matrix.Matrix1D, primes: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D;

    updateWeights(layer: ILayer, deltaWeights: matrix.Matrix2D, deltaBiases: matrix.Matrix1D, batchSize: number) {
        matrix.matrix1d_binary_in_place_op(layer.biases, deltaBiases, (b, d) => b + this.lr * d / batchSize);

        for (let j = 0; j < layer.size; j++) {
            matrix.matrix1d_binary_in_place_op(layer.weights[j], deltaWeights[j], (w, d) => {
                let l1Regularization = 0;
                if (layer.l1WeightRegularization > 0) {
                    l1Regularization = Math.sign(w) * layer.l1WeightRegularization
                }

                let l2Regularization = 0;
                if (layer.l2WeightRegularization > 0) {
                    l2Regularization = 2 * w * layer.l2WeightRegularization
                }

                const change = d / batchSize;
                return w - l1Regularization - l2Regularization + this.lr * change;
            });
        }
    }
}

class SgdOptimizer extends OptimizerBase {
    readonly description: string;
    readonly lr: number;

    private cache = new Map<ILayer, { tmp1: matrix.Matrix1D }>();

    constructor(lr = 0.01) {
        super();

        this.lr = lr;
        this.description = `sgd(lr: ${this.lr})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, primes: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                tmp1: matrix.zero(layer.size)
            })
        }

        const {tmp1} = this.cache.get(layer)!;
        matrix.matrix1d_binary_op(primes, error, (a, e) => layer.activation.moment(a) * e, tmp1);

        return tmp1;
    }
}

type NesterovCacheT = { tmp1: matrix.Matrix1D, weights: { moments: matrix.Matrix1D } };

class SgdNesterovOptimizer extends OptimizerBase {
    readonly description: string;
    readonly lr: number;
    readonly beta: number;

    private cache = new Map<ILayer, NesterovCacheT>();

    constructor(lr = 0.01, beta = 0.9) {
        super();

        this.lr = lr;
        this.beta = beta;

        this.description = `nesterov(beta: ${this.beta}, lr: ${this.lr})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, primes: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                tmp1: matrix.zero(layer.size),
                weights: {moments: matrix.zero(layer.size)}
            })
        }

        const s = this.cache.get(layer)!;

        // next gradient
        matrix.matrix1d_binary_op(primes, s.weights.moments, (a, m) => layer.activation.moment(a + m), s.tmp1);
        matrix.mul_to(s.tmp1, error);

        // calculate/update moment
        matrix.matrix1d_binary_in_place_op(s.weights.moments, s.tmp1, (m, g) => this.beta * m + (1 - this.beta) * g);

        // apply learning rate to moment
        matrix.matrix1d_binary_in_place_op(s.tmp1, s.weights.moments, (_, m) => m);

        return s.tmp1;
    }
}


type RMSPropCacheT = { velocities: matrix.Matrix1D, tmp1: matrix.Matrix1D };

class RMSPropOptimizer extends OptimizerBase {
    readonly description: string;
    readonly beta: number;
    readonly lr: number;
    readonly eps: number

    private cache = new Map<ILayer, RMSPropCacheT>();

    constructor(lr = 0.005, beta = 0.9, eps = 1e-8) {
        super();

        this.lr = lr;
        this.beta = beta;
        this.eps = eps;
        this.description = `rmsprop(beta: ${beta}, lr: ${this.lr}, eps: ${this.eps})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, primes: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                velocities: matrix.zero(layer.size),
                tmp1: matrix.zero(layer.size)
            })
        }

        const s = this.cache.get(layer)!;

        // next gradient
        matrix.matrix1d_binary_op(primes, error, (a, e) => e * layer.activation.moment(a), s.tmp1);

        matrix.matrix1d_binary_in_place_op(s.velocities, s.tmp1, (m, g) => this.beta * m + (1 - this.beta) * g * g);
        matrix.matrix1d_binary_in_place_op(s.tmp1, s.velocities, (g, m) => (1 / Math.sqrt(m + this.eps) * g));

        return s.tmp1;
    }
}

type AdamCacheT = {
    moments: matrix.Matrix1D,
    velocities: matrix.Matrix1D,
    tmp1: matrix.Matrix1D,
    tmp2: matrix.Matrix1D
};

class AdamOptimizer extends OptimizerBase {
    readonly description: string;
    readonly lr: number;
    readonly beta1: number;
    readonly beta2: number;
    readonly eps: number;

    private cache = new Map<ILayer, AdamCacheT>();

    constructor(lr = 0.005, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
        super();

        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;

        this.description = `adam(beta1: ${beta1}, beta2: ${beta2}, lr: ${this.lr}, eps: ${eps.toExponential()})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, primes: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
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
        matrix.matrix1d_binary_op(primes, error, (a, e) => layer.activation.moment(a) * e, s.tmp1);

        // moving smooth moment and velocity
        matrix.matrix1d_binary_in_place_op(s.moments, s.tmp1, (m, g) => m * this.beta1 + (1 - this.beta1) * g);
        matrix.matrix1d_binary_in_place_op(s.velocities, s.tmp1, (m, g) => m * this.beta2 + (1 - this.beta2) * g * g);

        // boost moment and velocity for first epochs
        matrix.matrix1d_binary_in_place_op(s.tmp1, s.moments, (_, m) => m / (1 - Math.pow(this.beta1, epoch + 1)));
        matrix.matrix1d_binary_in_place_op(s.tmp2, s.velocities, (_, v) => v / (1 - Math.pow(this.beta2, epoch + 1)));

        matrix.matrix1d_binary_in_place_op(s.tmp1, s.tmp2, (m, v) => m / (Math.sqrt(v) + this.eps));

        return s.tmp1;
    }

}

export type OptimizerT = "sgd" | "nesterov" | "adam" | "rmsprop";

export function buildOptimizer(optimizer: OptimizerT | IOptimizer = 'sgd') {
    const optimizer_param = typeof optimizer === "string" ? Optimizers[optimizer] : optimizer
    if (!optimizer_param) {
        throw new Error(`Unknown optimizer type ${optimizer_param}`);
    }

    if (typeof optimizer_param === "object") {
        return optimizer_param;
    }

    return new optimizer_param();
}

export const Optimizers = {
    sgd: SgdOptimizer,
    nesterov: SgdNesterovOptimizer,
    adam: AdamOptimizer,
    rmsprop: RMSPropOptimizer
}
