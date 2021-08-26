import * as matrix from "./matrix";
import * as iter from "./iter";

import {ILayer, IOptimizer} from "./base";
import {GlobalPool, MemorySlice} from "./memory";

class SgdOptimizer implements IOptimizer {
    readonly description: string;
    readonly lr: number;

    constructor(lr = 0.01) {
        this.lr = lr;
        this.description = `sgd(lr: ${this.lr})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.ManagedMatrix1D {
        const derivatives = layer.activation.moment(activations);
        const gradient = matrix.mul(error, derivatives);
        const result = matrix.matrix1d_unary_op(gradient, g => g * this.lr);

        derivatives.free();
        gradient.free();

        return result;
    }
}

class SgdNesterovOptimizer implements IOptimizer {
    readonly description: string;
    readonly lr: number;
    readonly beta: number;

    private cache = new WeakMap<ILayer, matrix.ManagedMatrix1D>();

    constructor(beta = 0.9, lr = 0.01) {
        this.lr = lr;
        this.beta = beta;

        this.description = `nesterov(beta: ${this.beta}, lr: ${this.lr})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.ManagedMatrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, matrix.zero(layer.size))
        }

        const moment = this.cache.get(layer)!;
        const nextValue = matrix.add(activations, moment);
        const derivatives = layer.activation.moment(nextValue);


        const gradient = matrix.mul(error, layer.activation.moment(derivatives));
        matrix.matrix1d_binary_in_place_op(moment, gradient, (m, g) => this.beta * m + this.lr * g);

        nextValue.free();
        derivatives.free();
        gradient.free();

        return moment;
    }
}

type AdamCacheT = { moments: matrix.Matrix1D, velocities: matrix.Matrix1D };

class AdamOptimizer implements IOptimizer {
    readonly description: string;
    readonly lr: number;
    readonly beta1: number;
    readonly beta2: number;
    readonly eps: number;

    private cache = new WeakMap<ILayer, AdamCacheT>();

    constructor(beta1 = 0.9, beta2 = 0.999, lr = 0.005, eps = 1e-8) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;

        this.description = `adam(beta1: ${beta1}, beta2: ${beta2}, lr: ${this.lr}, eps: ${eps.toExponential()})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.ManagedMatrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                moments: MemorySlice.from(iter.fill_value(0, layer.size)),
                velocities: MemorySlice.from(iter.fill_value(0, layer.size))
            })
        }

        const s = this.cache.get(layer)!;

        const derivatives = layer.activation.moment(activations);
        const gradient = matrix.mul(error, derivatives);

        matrix.matrix1d_binary_in_place_op(s.moments, gradient, (m, g) => m * this.beta1 + (1 - this.beta1) * g);
        matrix.matrix1d_binary_in_place_op(s.velocities, gradient, (m, g) => m * this.beta2 + (1 - this.beta2) * g * g);

        const mHats = GlobalPool.alloc(layer.size);
        const vHats = GlobalPool.alloc(layer.size);

        // @ts-ignore
        matrix.matrix1d_binary_in_place_op(mHats, s.moments, (_, m) => m / (1 - Math.pow(this.beta1, epoch + 1)));
        // @ts-ignore
        matrix.matrix1d_binary_in_place_op(vHats, s.velocities, (_, v) => v / (1 - Math.pow(this.beta2, epoch + 1)));
        // @ts-ignore
        matrix.matrix1d_binary_in_place_op(mHats, vHats, (m, v) => (m * this.lr) / (Math.sqrt(v) + this.eps));

        derivatives.free();
        gradient.free();
        vHats.free();

        return mHats;
    }

}

export type OptimizerT = "sgd" | "nesterov" | "adam";

export const Optimizers = {
    sgd: SgdOptimizer,
    nesterov: SgdNesterovOptimizer,
    adam: AdamOptimizer
}
