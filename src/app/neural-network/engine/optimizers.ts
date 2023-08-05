import * as matrix from "./matrix";
import {Matrix1D, Matrix2D} from "./matrix";
import {ILayer, IOptimizer} from "./base";

abstract class OptimizerBase implements IOptimizer {
    abstract readonly description: string;
    readonly decay: number;

    private _cache = new Map<ILayer, { tmp1: Matrix1D }>();
    private readonly _initialLearningRate: number;
    private _lr: number;


    get lr(): number {return this._lr};


    protected constructor(lr: number, decay: number) {
        this._initialLearningRate = lr;
        this._lr = lr;
        this.decay = decay;
    }

    beforePass() {}

    afterPass() {
        if (this.decay > 0) {
            this._lr = (this._initialLearningRate * this._lr) / (this._initialLearningRate + this.decay * this._lr);
        }
    }

    step(layer: ILayer, activations: Matrix1D, primes: Matrix1D, error: Matrix1D, epoch: number): Matrix1D {
        if (!this._cache.has(layer)) {
            this._cache.set(layer, {
                tmp1: matrix.zero(layer.size)
            })
        }

        const {tmp1} = this._cache.get(layer)!;
        matrix.matrix1d_binary_op(primes, error, (a, e) => layer.activation.moment(a) * e, tmp1);

        return tmp1;
    }

    updateWeights(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, epoch: number, batchSize: number) {
        matrix.matrix1d_binary_in_place_op(layer.biases, deltaBiases, (b, dB) => {
            let L1Regularization = 0;
            if (layer.l1BiasRegularization > 0) {
                L1Regularization = Math.sign(b) * layer.l1BiasRegularization
            }

            let l2Regularization = 0;
            if (layer.l2BiasRegularization > 0) {
                l2Regularization = 2 * b * layer.l2BiasRegularization
            }

            return b - this.lr * (L1Regularization + l2Regularization + dB / batchSize)
        });

        for (let j = 0; j < layer.size; j++) {
            matrix.matrix1d_binary_in_place_op(layer.weights[j], deltaWeights[j], (w, dW) => {
                let L1Regularization = 0;
                if (layer.l1WeightRegularization > 0) {
                    L1Regularization = Math.sign(w) * layer.l1WeightRegularization
                }

                let l2Regularization = 0;
                if (layer.l2WeightRegularization > 0) {
                    l2Regularization = 2 * w * layer.l2WeightRegularization
                }

                return w - this.lr * (L1Regularization + l2Regularization + dW / batchSize);
            });
        }
    }
}

class SgdOptimizer extends OptimizerBase {
    readonly description: string;

    constructor(lr = 1, decay = 0) {
        super(lr, decay);

        this.description = `sgd(lr: ${this.lr})`;
    }
}

type SgdMomentumCacheT = { mWeights: Matrix2D, mBiases: Matrix1D };

class SgdMomentumOptimizer extends OptimizerBase {
    private cache = new Map<ILayer, SgdMomentumCacheT>();

    readonly description: string;

    constructor(lr = 0.01, decay = 0, readonly beta = 0.5) {
        super(lr, decay);

        this.description = `sgd-momentum(beta: ${beta}, lr: ${this.lr})`;
    }

    updateWeights(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, epoch: number, batchSize: number) {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                mWeights: matrix.zero_2d(layer.size, layer.prevSize),
                mBiases: matrix.zero(layer.size)
            })
        }

        const {mWeights, mBiases} = this.cache.get(layer)!;

        matrix.matrix2d_binary_in_place_op(mWeights, deltaWeights, (mW, dW) => dW - this.beta * mW);
        matrix.matrix1d_binary_in_place_op(mBiases, deltaBiases, (mB, dB) => dB - this.beta * mB);

        super.updateWeights(layer, mWeights, mBiases, epoch, batchSize);
    }
}

type NesterovCacheT = { tmp1: Matrix1D, momentum: Matrix1D };

class SgdNesterovOptimizer extends OptimizerBase {
    readonly description: string;
    readonly beta: number;

    private cache = new Map<ILayer, NesterovCacheT>();

    constructor(lr = 0.01, decay = 0, beta = 0.9) {
        super(lr, decay);

        this.beta = beta;
        this.description = `nesterov(beta: ${this.beta}, lr: ${this.lr})`;
    }

    step(layer: ILayer, activations: matrix.Matrix1D, primes: matrix.Matrix1D, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                tmp1: matrix.zero(layer.size),
                momentum: matrix.zero(layer.size)
            })
        }

        const s = this.cache.get(layer)!;

        // next gradient
        matrix.matrix1d_binary_op(primes, s.momentum, (a, m) => layer.activation.moment(a + m), s.tmp1);
        matrix.mul_to(s.tmp1, error);

        // calculate/update moment
        matrix.matrix1d_binary_in_place_op(s.momentum, s.tmp1, (m, g) => this.beta * m + (1 - this.beta) * g);

        return s.tmp1;
    }
}


type RMSPropCacheT = { mWeights: Matrix2D, mBiases: Matrix1D };

class RMSPropOptimizer extends OptimizerBase {
    readonly description: string;
    readonly beta: number;
    readonly eps: number

    private cache = new Map<ILayer, RMSPropCacheT>();

    constructor(lr = 0.001, decay = 0, beta = 0.9, eps = 1e-8) {
        super(lr, decay);

        this.beta = beta;
        this.eps = eps;
        this.description = `rmsprop(beta: ${beta}, lr: ${this.lr}, eps: ${this.eps})`;
    }

    updateWeights(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, epoch: number, batchSize: number) {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                mWeights: matrix.zero_2d(layer.size, layer.prevSize),
                mBiases: matrix.zero(layer.size)
            })
        }

        const {mWeights, mBiases} = this.cache.get(layer)!;

        matrix.matrix2d_binary_in_place_op(mWeights, deltaWeights, (mW, dW) =>
            this.beta * mW + (1 - this.beta) * Math.pow(dW, 2));
        matrix.matrix1d_binary_in_place_op(mBiases, deltaBiases, (mB, dB) =>
            this.beta * mB + (1 - this.beta) * Math.pow(dB, 2));

        matrix.matrix2d_binary_in_place_op(deltaWeights, mWeights, (dW, mW) =>
            dW / (Math.sqrt(mW) + this.eps));
        matrix.matrix1d_binary_in_place_op(deltaBiases, mBiases, (dB, mB) =>
            dB / (Math.sqrt(mB) + this.eps));

        super.updateWeights(layer, deltaWeights, deltaBiases, epoch, batchSize);
    }
}

type AdamCacheT = {
    mWeights: Matrix2D, mBiases: Matrix1D,
    cWeights: Matrix2D, cBiases: Matrix1D
};

class AdamOptimizer extends OptimizerBase {
    readonly description: string;
    readonly beta1: number;
    readonly beta2: number;
    readonly eps: number;

    private cache = new Map<ILayer, AdamCacheT>();

    constructor(lr = 0.001, decay = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
        super(lr, decay);

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;

        this.description = `adam(beta1: ${beta1}, beta2: ${beta2}, lr: ${this.lr}, eps: ${eps.toExponential()})`;
    }

    updateWeights(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, epoch: number, batchSize: number) {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                mWeights: matrix.zero_2d(layer.size, layer.prevSize),
                mBiases: matrix.zero(layer.size),
                cWeights: matrix.zero_2d(layer.size, layer.prevSize),
                cBiases: matrix.zero(layer.size)
            });
        }

        const {
            mWeights, mBiases,
            cWeights, cBiases
        } = this.cache.get(layer)!;

        matrix.matrix2d_binary_in_place_op(mWeights, deltaWeights, (mW, dW) =>
            this.beta1 * mW + (1 - this.beta1) * dW);
        matrix.matrix1d_binary_in_place_op(mBiases, deltaBiases, (mB, dB) =>
            this.beta1 * mB + (1 - this.beta1) * dB);

        matrix.matrix2d_binary_in_place_op(cWeights, deltaWeights, (cW, dW) =>
            this.beta2 * cW + (1 - this.beta2) * Math.pow(dW, 2));
        matrix.matrix1d_binary_in_place_op(cBiases, deltaBiases, (cB, dB) =>
            this.beta2 * cB + (1 - this.beta2) * Math.pow(dB, 2));

        const beta1Ep = Math.pow(this.beta1, epoch + 1);
        const beta2Ep = Math.pow(this.beta2, epoch + 1);

        matrix.matrix2d_binary_op(mWeights, cWeights, (mW, cW) =>
                (mW / (1 - beta1Ep)) / (Math.sqrt(cW / (1 - beta2Ep)) + this.eps),
            deltaWeights);

        matrix.matrix1d_binary_op(mBiases, cBiases, (mB, cB) =>
                (mB / (1 - beta1Ep)) / (Math.sqrt(cB / (1 - beta2Ep)) + this.eps),
            deltaBiases);

        super.updateWeights(layer, deltaWeights, deltaBiases, epoch, batchSize);
    }
}

export type OptimizerT = "sgd" | "sgdMomentum" | "nesterov" | "adam" | "rmsprop";

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
    sgdMomentum: SgdMomentumOptimizer,
    nesterov: SgdNesterovOptimizer,
    adam: AdamOptimizer,
    rmsprop: RMSPropOptimizer
}
