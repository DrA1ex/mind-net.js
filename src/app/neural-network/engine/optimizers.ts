import {Param} from "../serialization";
import {ILayer, IOptimizer} from "./base";
import {Matrix1D, Matrix2D} from "./matrix";
import * as matrix from "./matrix";

export abstract class OptimizerBase implements IOptimizer {
    abstract readonly description: string;

    @Param()
    readonly decay: number;

    @Param()
    get lr(): number {return this._lr};

    private _cache = new Map<ILayer, { tmp1: Matrix1D }>();
    private readonly _initialLearningRate: number;
    private _lr: number;


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

    step(layer: ILayer, error: Matrix1D, epoch: number): Matrix1D {
        if (!this._cache.has(layer)) {
            this._cache.set(layer, {
                tmp1: matrix.zero(layer.size)
            })
        }

        const {tmp1} = this._cache.get(layer)!;

        layer.activation.moment(layer.output, tmp1);
        matrix.matrix1d_binary_in_place_op(tmp1, error, (a, e) => a * e);

        return tmp1;
    }

    updateWeights(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, epoch: number, batchSize: number) {
        this.applyBatchSize(layer, deltaWeights, deltaBiases, batchSize);
        this.applyRegularization(layer, deltaWeights, deltaBiases);
        this.applyDelta(layer, deltaWeights, deltaBiases);
    }

    protected applyBatchSize(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, batchSize: number) {
        if (batchSize === 1) return;

        matrix.matrix2d_unary_in_place_op(deltaWeights, (dW) => dW / batchSize);
        matrix.matrix1d_unary_in_place_op(deltaBiases, (dB) => dB / batchSize);
    }

    protected applyRegularization(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D) {
        if (layer.l1BiasRegularization > 0 || layer.l2BiasRegularization > 0) {
            matrix.matrix1d_binary_in_place_op(deltaBiases, layer.biases, (dB, b) => {
                let L1Regularization = 0;
                if (layer.l1BiasRegularization > 0) {
                    L1Regularization = Math.sign(b) * layer.l1BiasRegularization
                }

                let l2Regularization = 0;
                if (layer.l2BiasRegularization > 0) {
                    l2Regularization = 2 * b * layer.l2BiasRegularization
                }

                return dB + (L1Regularization + l2Regularization);
            });
        }

        if (layer.l1WeightRegularization > 0 || layer.l2WeightRegularization > 0) {
            matrix.matrix2d_binary_in_place_op(deltaWeights, layer.weights, (dW, w) => {
                let L1Regularization = 0;
                if (layer.l1WeightRegularization > 0) {
                    L1Regularization = Math.sign(w) * layer.l1WeightRegularization
                }

                let l2Regularization = 0;
                if (layer.l2WeightRegularization > 0) {
                    l2Regularization = 2 * w * layer.l2WeightRegularization
                }

                return dW + (L1Regularization + l2Regularization);
            });
        }
    }

    protected applyDelta(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D) {
        matrix.matrix1d_binary_in_place_op(layer.biases, deltaBiases, (b, dB) => {
            return b - this.lr * dB;
        });

        matrix.matrix2d_binary_in_place_op(layer.weights, deltaWeights, (w, dW) => {
            return w - this.lr * dW;
        });
    }
}

export type MomentCacheT = { [key: string]: any };

export abstract class AbstractMomentAcceleratedOptimizer<TCache extends MomentCacheT> extends OptimizerBase {
    public moments = new Map<ILayer, TCache>();

    abstract updateAveragedWeights(
        layer: ILayer,
        deltaWeights: Matrix2D,
        deltaBiases: Matrix1D,
        epoch: number
    ): { dW: Matrix2D, dB: Matrix1D };

    updateWeights(
        layer: ILayer,
        deltaWeights: Matrix2D,
        deltaBiases: Matrix1D,
        epoch: number,
        batchSize: number
    ) {
        // Apply delta averaging to ensure that any batch size equally affects the moment
        this.applyBatchSize(layer, deltaWeights, deltaBiases, batchSize);
        this.applyRegularization(layer, deltaWeights, deltaBiases);

        const {dW, dB} = this.updateAveragedWeights(layer, deltaWeights, deltaBiases, epoch);

        this.applyDelta(layer, dW, dB);
    }
}

type SgdCtorArgsT = {
    lr: number;
    decay: number;
};

const DefaultSgdArgs: SgdCtorArgsT = {
    lr: 1,
    decay: 0,
};

export class SgdOptimizer extends OptimizerBase {
    readonly description: string;

    constructor(options: Partial<SgdCtorArgsT> = DefaultSgdArgs) {
        const {lr, decay}: SgdCtorArgsT = {...DefaultSgdArgs, ...options};
        super(lr, decay);

        this.description = `sgd(lr: ${this.lr})`;
    }
}

type SgdMomentumCtorArgsT = {
    lr: number;
    decay: number;
    beta: number;
};

const DefaultSgdMomentumArgs: SgdMomentumCtorArgsT = {
    lr: 0.01,
    decay: 0,
    beta: 0.5,
};

type SgdMomentumCacheT = { mWeights: Matrix2D, mBiases: Matrix1D, meta: { callCount: number } };

export class SgdMomentumOptimizer extends AbstractMomentAcceleratedOptimizer<SgdMomentumCacheT> {
    readonly description: string;

    @Param()
    beta: number;

    constructor(options: Partial<SgdMomentumCtorArgsT> = DefaultSgdMomentumArgs) {
        const {lr, decay, beta}: SgdMomentumCtorArgsT = {...DefaultSgdMomentumArgs, ...options};
        super(lr, decay);

        this.beta = beta;
        this.description = `sgd-momentum(beta: ${beta}, lr: ${this.lr})`;
    }

    updateAveragedWeights(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, epoch: number) {
        if (!this.moments.has(layer)) {
            this.moments.set(layer, {
                mWeights: matrix.zero_2d(layer.size, layer.prevSize),
                mBiases: matrix.zero(layer.size),
                meta: {callCount: 0}
            })
        }

        const {mWeights, mBiases, meta} = this.moments.get(layer)!;

        // In the original formula, we add the moment to the weights, but in other optimizers, we subtract it.
        // To achieve this in our optimizer, we need to copy the moment and negate it.
        // However, this approach can degrade performance.
        // To avoid degradation, we modify the formula to return the already negated value.
        // But this modification also affects the moment by making it negated.
        // To revert the negation of the moment, we should negate it in the subsequent iterations.
        //
        // Mathematically, for each epoch:
        // 0 call: mW_1 = (beta * mW_0 - dW), we modify this to: mW_1 = -(beta * mW_0 - dW)
        // 1 call: mW_2 = (beta * mW_1 - dW), we modify this to: mW_2 = -(beta * mW_1 - dW)
        //          We further simplify: mW_2 = -(beta * -mW_1 - dW) = (beta * mW_1 + dW)
        // 3 call: mW_2 = -(beta * -mW_2 - dW), which simplifies to: (beta * mW_2 + dW) and so on

        matrix.matrix2d_binary_in_place_op(mWeights, deltaWeights, (mW, dW) => {
            if (meta.callCount === 0) {
                return dW;
            } else {
                return -(this.beta * -mW - dW)
            }
        });

        matrix.matrix1d_binary_in_place_op(mBiases, deltaBiases, (mB, dB) => {
            if (meta.callCount === 0) {
                return dB;
            } else {
                return -(this.beta * -mB - dB)
            }
        });

        matrix.matrix2d_binary_in_place_op(deltaWeights, mWeights, (dW, mW) => {
            return mW;
        });
        matrix.matrix1d_binary_in_place_op(deltaBiases, mBiases, (dB, mB) => {
            return mB;
        });

        ++meta.callCount;
        return {dW: deltaWeights, dB: deltaBiases};
    }
}

type SgdNesterovCtorArgsT = {
    lr: number;
    decay: number;
    beta: number;
};

const DefaultSgdNesterovArgs: SgdNesterovCtorArgsT = {
    lr: 0.01,
    decay: 0,
    beta: 0.9,
};

type NesterovCacheT = { tmp1: Matrix1D, nextGrad: Matrix1D, momentum: Matrix1D };

export class SgdNesterovOptimizer extends OptimizerBase {
    readonly description: string;

    @Param()
    beta: number;

    private cache = new Map<ILayer, NesterovCacheT>();

    constructor(options: Partial<SgdNesterovCtorArgsT> = DefaultSgdNesterovArgs) {
        const {lr, decay, beta}: SgdNesterovCtorArgsT = {...DefaultSgdNesterovArgs, ...options};
        super(lr, decay);

        this.beta = beta;
        this.description = `nesterov(beta: ${this.beta}, lr: ${this.lr})`;
    }

    step(layer: ILayer, error: matrix.Matrix1D, epoch: number): matrix.Matrix1D {
        if (!this.cache.has(layer)) {
            this.cache.set(layer, {
                tmp1: matrix.zero(layer.size),
                nextGrad: matrix.zero(layer.size),
                momentum: matrix.zero(layer.size)
            })
        }

        const s = this.cache.get(layer)!;

        // next gradient
        matrix.add(layer.output, s.momentum, s.nextGrad);
        layer.activation.moment(s.nextGrad, s.tmp1);
        matrix.mul_to(error, s.tmp1);

        // calculate/update moment
        matrix.matrix1d_binary_in_place_op(s.momentum, s.tmp1, (m, g) => this.beta * m + (1 - this.beta) * g);

        return s.tmp1;
    }
}

type RMSPropCtorArgsT = {
    lr: number;
    decay: number;
    beta: number;
    eps: number;
};

const DefaultRMSPropArgs: RMSPropCtorArgsT = {
    lr: 0.001,
    decay: 0,
    beta: 0.9,
    eps: 1e-7,
};

type RMSPropCacheT = { mWeights: Matrix2D, mBiases: Matrix1D };

export class RMSPropOptimizer extends AbstractMomentAcceleratedOptimizer<RMSPropCacheT> {
    readonly description: string;

    @Param()
    beta: number;

    @Param()
    eps: number

    constructor(options: Partial<RMSPropCtorArgsT> = DefaultRMSPropArgs) {
        const {
            lr, decay, beta, eps,
        }: RMSPropCtorArgsT = {...DefaultRMSPropArgs, ...options};
        super(lr, decay);

        this.beta = beta;
        this.eps = eps;
        this.description = `rmsprop(beta: ${beta}, lr: ${this.lr}, eps: ${this.eps})`;
    }

    updateAveragedWeights(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, epoch: number) {
        if (!this.moments.has(layer)) {
            this.moments.set(layer, {
                mWeights: matrix.zero_2d(layer.size, layer.prevSize),
                mBiases: matrix.zero(layer.size)
            })
        }

        const {mWeights, mBiases} = this.moments.get(layer)!;

        matrix.matrix2d_binary_in_place_op(mWeights, deltaWeights, (mW, dW) =>
            this.beta * mW + (1 - this.beta) * Math.pow(dW, 2));
        matrix.matrix1d_binary_in_place_op(mBiases, deltaBiases, (mB, dB) =>
            this.beta * mB + (1 - this.beta) * Math.pow(dB, 2));

        matrix.matrix2d_binary_in_place_op(deltaWeights, mWeights, (dW, mW) =>
            dW / (Math.sqrt(mW) + this.eps));
        matrix.matrix1d_binary_in_place_op(deltaBiases, mBiases, (dB, mB) =>
            dB / (Math.sqrt(mB) + this.eps));

        return {dW: deltaWeights, dB: deltaBiases};
    }
}

type AdamCtorArgsT = {
    lr: number;
    decay: number;
    beta1: number;
    beta2: number;
    eps: number;
};

const DefaultAdamArgs: AdamCtorArgsT = {
    lr: 0.001,
    decay: 0,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-7,
};

type AdamCacheT = {
    mWeights: Matrix2D, mBiases: Matrix1D,
    cWeights: Matrix2D, cBiases: Matrix1D
};

export class AdamOptimizer extends AbstractMomentAcceleratedOptimizer<AdamCacheT> {
    readonly description: string;

    @Param()
    beta1: number;

    @Param()
    beta2: number;

    @Param()
    eps: number;

    constructor(options: Partial<AdamCtorArgsT> = DefaultAdamArgs) {
        const {
            lr, decay, beta1, beta2, eps,
        }: AdamCtorArgsT = {...DefaultAdamArgs, ...options};
        super(lr, decay);

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;

        this.description = `adam(beta1: ${beta1}, beta2: ${beta2}, lr: ${this.lr}, eps: ${eps.toExponential()})`;
    }

    updateAveragedWeights(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, epoch: number) {
        if (!this.moments.has(layer)) {
            this.moments.set(layer, {
                mWeights: matrix.zero_2d(layer.size, layer.prevSize),
                mBiases: matrix.zero(layer.size),
                cWeights: matrix.zero_2d(layer.size, layer.prevSize),
                cBiases: matrix.zero(layer.size)
            });
        }

        const {
            mWeights, mBiases,
            cWeights, cBiases
        } = this.moments.get(layer)!;

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

        return {dW: deltaWeights, dB: deltaBiases};
    }
}

export type OptimizerT = "sgd" | "sgdMomentum" | "nesterov" | "adam" | "rmsprop";

export function buildOptimizer(optimizer: OptimizerT | IOptimizer = 'sgd') {
    const optimizer_param = typeof optimizer === "string" ? OptimizersMap[optimizer] : optimizer
    if (!optimizer_param) {
        throw new Error(`Unknown optimizer type ${optimizer_param}`);
    }

    if (typeof optimizer_param === "object") {
        return optimizer_param;
    }

    return new optimizer_param();
}

export const OptimizersMap = {
    sgd: SgdOptimizer,
    sgdMomentum: SgdMomentumOptimizer,
    nesterov: SgdNesterovOptimizer,
    adam: AdamOptimizer,
    rmsprop: RMSPropOptimizer
}

export const Optimizers = {
    SgdOptimizer,
    SgdMomentumOptimizer,
    SgdNesterovOptimizer,
    AdamOptimizer,
    RMSPropOptimizer,
}
