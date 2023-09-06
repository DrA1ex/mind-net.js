import * as Iter from "../iter";
import * as Matrix from "../matrix";

import {ILayer, IOptimizer, ILoss, IModel, ModelTrainOptionsT} from "../base";
import {Matrix1D, Matrix2D} from "../matrix";
import {buildLoss, CategoricalCrossEntropyLoss, LossT} from "../loss";
import {buildOptimizer, OptimizerT} from "../optimizers";
import {SoftMaxActivation} from "../activations";
import {Color, ValueLimit} from "../../utils/progress";
import {ProgressFn} from "../../utils/fetch";
import {ProgressUtils} from "../../neural-network";

export type NeuralNetworkSnapshot = {
    weights: Matrix2D[],
    biases: Matrix1D[]
};

export type LayerCache = {
    deltaWeights: Matrix2D,
    deltaBiases: Matrix1D,
    mask: Matrix1D,
};

export const DefaultTrainOpts: ModelTrainOptionsT = {
    batchSize: 32,
    epochs: 1,
    progress: true,
    progressOptions: {
        update: (typeof process !== "undefined"),
        color: Color.yellow,
        limit: ValueLimit.inclusive,
        progressThrottle: 500
    }
}

export abstract class ModelBase implements IModel {
    protected _epoch: number = 0;

    protected compiled: boolean = false;
    protected cache = new Map<ILayer, LayerCache>();
    protected lossErrorCache!: Matrix1D;

    readonly optimizer: IOptimizer;
    readonly loss: ILoss;

    get isCompiled() {return this.compiled;}
    get inputSize() {return this.layers[0].size;}
    get outputSize() {return this.layers[this.layers.length - 1].size;}

    public get epoch() {
        return this._epoch;
    }

    abstract readonly layers: ILayer[];
    abstract isTrainable(layer: ILayer): boolean;

    constructor(optimizer: OptimizerT | IOptimizer = 'sgd', loss: LossT | ILoss = "mse") {
        this.optimizer = buildOptimizer(optimizer);
        this.loss = buildLoss(loss);
    }

    compute(input: Matrix1D): Matrix1D {
        this._assertCompiled();
        this._assertInputSize(input);

        const output = this._forward(input);
        return Matrix.copy(output);
    }

    evaluate(input: Matrix1D[], expected: Matrix1D[]) {
        this._assertInputSize2d(input);
        this._assertExpectedSize2d(expected);
        this._assertInputOutputSize(input, expected);

        const predicated = input.map((data) => this.compute(data));
        for (const data of expected) {
            this._assertExpectedSize(data);
        }

        return {
            loss: this.loss.loss(predicated, expected),
            accuracy: this.loss.accuracy(predicated, expected)
        };
    }

    train(
        input: Matrix1D[], expected: Matrix1D[], options: Partial<ModelTrainOptionsT> = {}
    ) {
        this._assertCompiled();
        this._assertInputSize2d(input);
        this._assertExpectedSize2d(expected);
        this._assertInputOutputSize(input, expected);

        const opts = {...DefaultTrainOpts, ...options};

        const batchCtrl = opts.progress
            ? ProgressUtils.progressBatchCallback(input.length, opts.epochs, opts.progressOptions)
            : undefined;

        for (let i = 0; i < opts.epochs; i++) {
            this.beforeTrain();

            const shuffledTrainSet = Iter.shuffle(Array.from(Iter.zip(input, expected)));
            for (const batch of Iter.partition(shuffledTrainSet, opts.batchSize)) {
                this.trainBatch(batch, batchCtrl?.progressFn);
                batchCtrl?.add(batch.length);
            }

            this.afterTrain();
        }
    }

    public trainBatch(batch: Iterable<[Matrix1D, Matrix1D]>, progressFn?: ProgressFn) {
        this._assertCompiled();

        this._clearDelta();

        const totalCount = (batch as any).length ?? 0;
        let count = 0;
        for (const [trainInput, trainExpected] of batch) {
            if (progressFn) progressFn(count, totalCount);

            this._assertInputSize(trainInput);
            this._assertExpectedSize(trainExpected);

            const output = this._forward(trainInput, true);
            const loss = this.loss.calculateError(output, trainExpected, this.lossErrorCache);

            this._backprop(loss);
            ++count;
        }

        if (progressFn) progressFn(count, totalCount);

        this._applyDelta(count);
    }

    public beforeTrain() {
        this.optimizer.beforePass();
    }

    public afterTrain() {
        this.optimizer.afterPass();
        this._epoch += 1;
    }

    compile(allowMultipleLayerUsage = false): void {
        if (this.compiled) {
            return;
        }

        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            if (layer.activation instanceof SoftMaxActivation) {
                if (i !== this.layers.length - 1) {
                    throw new Error("SoftMax activation supported only for last layer");
                } else if (!(this.loss instanceof CategoricalCrossEntropyLoss)) {
                    throw new Error("SoftMax activation supported only with CategoricalCrossEntropy loss");
                }
            }

            const prevSize = i > 0 ? this.layers[i - 1].size : 0;
            layer.build(i, prevSize, allowMultipleLayerUsage);
            this.cache.set(layer, {
                deltaWeights: Matrix.zero_2d(layer.size, prevSize),
                deltaBiases: Matrix.zero(layer.size),
                mask: Matrix.one(layer.size),
            });
        }

        this.lossErrorCache = Matrix.zero(this.layers[this.layers.length - 1].size);

        this.compiled = true;
    }

    protected _forward(input: Matrix1D, isTraining = false): Matrix1D {
        let result = input;
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];

            const prime = layer.step(result);
            result = layer.activation.forward(prime, layer.activationOutput);

            if (isTraining && layer.dropout > 0) {
                const mask = this.cache.get(layer)!.mask;
                this._calculateDropoutMask(layer, mask);
                this._applyDropoutMask(layer.activationOutput, mask);
            }
        }

        return result;
    }

    protected _calculateDropoutMask(layer: ILayer, dst: Matrix1D) {
        const rate = 1 - layer.dropout;
        const scale = 1 / rate;
        const maxZeros = Math.floor(layer.size * layer.dropout);

        let count = 0;
        for (let i = 0; i < layer.size; i++) {
            if (count < maxZeros && Math.random() >= rate) {
                dst[i] = 0;
                count++;
            } else {
                dst[i] = scale;
            }
        }
    }

    protected _applyDropoutMask(values: Matrix1D, mask: Matrix1D) {
        Matrix.mul_to(mask, values);
    }

    protected _backprop(loss: Matrix1D) {
        let errors = loss;

        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];
            const {deltaWeights, deltaBiases, mask} = this.cache.get(layer)!;

            if (layer.dropout > 0) {
                this._applyDropoutMask(errors, mask);
            }

            const gradient = this.optimizer.step(layer, errors, this.epoch);
            errors = layer.backward(gradient, deltaWeights, deltaBiases);
        }
    }

    protected _clearDelta(): void {
        for (const layer of this.layers) {
            const {deltaWeights, deltaBiases} = this.cache.get(layer)!;

            deltaBiases.fill(0);
            deltaWeights.forEach(w => w.fill(0));
        }
    }

    protected _applyDelta(batchSize: number): void {
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const {deltaWeights, deltaBiases} = this.cache.get(layer)!;

            this._applyLayerDelta(layer, deltaWeights, deltaBiases, batchSize);
        }
    }

    protected _applyLayerDelta(layer: ILayer, deltaWeights: Matrix2D, deltaBiases: Matrix1D, batchSize: number): void {
        if (!this.isTrainable(layer)) return;

        this.optimizer.updateWeights(layer, deltaWeights, deltaBiases, this.epoch, batchSize);
    }

    getSnapshot(): NeuralNetworkSnapshot {
        return {
            weights: this.layers.slice(1).map(l => l.weights.map(w => Matrix.copy(w))),
            biases: this.layers.slice(1).map(l => Matrix.copy(l.biases))
        };
    }

    protected _assertCompiled() {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }
    }

    protected _assertInputSize(input: Matrix1D) {
        if (!input) {
            throw new Error("Input data missing or in wrong format");
        }

        const inSize = this.layers[0].size;
        if (input.length !== inSize) {
            throw new Error(`Input matrix has different size. Expected size ${inSize}, got ${input.length}`);
        }
    }

    protected _assertInputSize2d(input: Matrix2D) {
        if (!input || !input[0]) {
            throw new Error("Input data missing or in wrong format");
        }

        const inSize = this.layers[0].size;
        if (input[0].length !== inSize) {
            throw new Error(`Input matrix has different size. Expected size ${inSize}, got ${input[0].length}`);
        }
    }

    protected _assertExpectedSize(expected: Matrix1D) {
        if (!expected) {
            throw new Error("Expected data missing or in wrong format");
        }

        const outSize = this.layers[this.layers.length - 1].size;
        if (expected.length !== outSize) {
            throw new Error(`Expected matrix has different size. Expected size ${outSize}, got ${expected.length}`);
        }
    }

    protected _assertExpectedSize2d(expected: Matrix2D) {
        if (!expected || !expected[0]) {
            throw new Error("Expected data missing or in wrong format");
        }

        const outSize = this.layers[this.layers.length - 1].size;
        if (expected[0].length !== outSize) {
            throw new Error(`Expected matrix has different size. Expected size ${outSize}, got ${expected[0].length}`);
        }
    }

    protected _assertInputOutputSize(input: Matrix2D, expected: Matrix2D) {
        if (input.length !== expected.length) {
            throw new Error(`Inconsistent data length: input length ${input.length} != expected length ${expected.length}`)
        }
    }
}