import * as iter from "../iter";
import * as matrix from "../matrix";

import {ILayer, IOptimizer, ILoss, IModel, ModelTrainOptionsT} from "../base";
import {buildLoss, LossT} from "../loss";
import {buildOptimizer, OptimizerT} from "../optimizers";

export type NeuralNetworkSnapshot = {
    weights: matrix.Matrix2D[],
    biases: matrix.Matrix1D[]
};

export type LayerCache = {
    deltaWeights: matrix.Matrix2D,
    deltaBiases: matrix.Matrix1D,
    mask: matrix.Matrix1D,
};

export abstract class ModelBase implements IModel {
    protected _epoch: number = 0;

    protected compiled: boolean = false;
    protected cache = new Map<ILayer, LayerCache>();

    private _lossErrorCache!: matrix.Matrix1D;

    readonly optimizer: IOptimizer;
    readonly loss: ILoss;

    get inputSize() {return this.layers[0].size;}
    get outputSize() {return this.layers[this.layers.length - 1].size;}

    public get epoch() {
        return this._epoch;
    }

    abstract readonly layers: ILayer[];

    abstract compile(...args: any[]): void

    constructor(optimizer: OptimizerT | IOptimizer = 'sgd', loss: LossT | ILoss = "mse") {
        this.optimizer = buildOptimizer(optimizer);
        this.loss = buildLoss(loss);
    }

    compute(input: matrix.Matrix1D): matrix.Matrix1D {
        this._assertCompiled();
        this._assertInputSize(input);

        const output = this._forward(input);
        return matrix.copy(output);
    }

    evaluate(input: matrix.Matrix1D[], expected: matrix.Matrix1D[]) {
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
        input: matrix.Matrix1D[], expected: matrix.Matrix1D[],
        {batchSize = 32, epochs = 1}: Partial<ModelTrainOptionsT> = {}
    ) {
        this._assertCompiled();
        this._assertInputSize2d(input);
        this._assertExpectedSize2d(expected);
        this._assertInputOutputSize(input, expected);

        for (let i = 0; i < epochs; i++) {
            this.beforeTrain();

            const shuffledTrainSet = iter.shuffle(Array.from(iter.zip(input, expected)));
            for (const batch of iter.partition(shuffledTrainSet, batchSize)) {
                this.trainBatch(batch);
            }

            this.afterTrain();
        }
    }

    public trainBatch(batch: Iterable<[matrix.Matrix1D, matrix.Matrix1D]>) {
        this._assertCompiled();

        this._clearDelta();

        //TODO: Refactor
        if (!this._lossErrorCache) this._lossErrorCache = matrix.zero(this.layers[this.layers.length - 1].size);

        let count = 0;
        for (const [trainInput, trainExpected] of batch) {
            this._assertInputSize(trainInput);
            this._assertExpectedSize(trainExpected);

            const output = this._forward(trainInput, true);
            const loss = this.loss.calculateError(output, trainExpected, this._lossErrorCache);

            this._backprop(loss);
            ++count;
        }

        this._applyDelta(count);
    }

    public beforeTrain() {
        this.optimizer.beforePass();
    }

    public afterTrain() {
        this.optimizer.afterPass();
        this._epoch += 1;
    }

    protected _forward(input: matrix.Matrix1D, isTraining = false): matrix.Matrix1D {
        let result = input;
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];

            const prime = layer.step(result);
            result = layer.activation.value(prime, layer.activationOutput);

            if (isTraining && layer.dropout > 0) {
                const mask = this.cache.get(layer)!.mask;
                this._calculateDropoutMask(layer, mask);
                this._applyDropoutMask(layer.activationOutput, mask);
            }
        }

        return result;
    }

    protected _calculateDropoutMask(layer: ILayer, dst: matrix.Matrix1D) {
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

    protected _applyDropoutMask(values: matrix.Matrix1D, mask: matrix.Matrix1D) {
        matrix.mul_to(values, mask);
    }

    protected _backprop(loss: matrix.Matrix1D) {
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
            this._applyLayerDelta(layer, batchSize);
        }
    }

    protected _applyLayerDelta(layer: ILayer, batchSize: number): void {
        const {deltaWeights, deltaBiases} = this.cache.get(layer)!;
        this.optimizer.updateWeights(layer, deltaWeights, deltaBiases, this.epoch, batchSize);
    }

    getSnapshot(): NeuralNetworkSnapshot {
        return {
            weights: this.layers.slice(1).map(l => l.weights.map(w => matrix.copy(w))),
            biases: this.layers.slice(1).map(l => matrix.copy(l.biases))
        };
    }

    protected _assertCompiled() {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }
    }

    protected _assertInputSize(input: matrix.Matrix1D) {
        if (!input) {
            throw new Error("Input data missing or in wrong format");
        }

        const inSize = this.layers[0].size;
        if (input.length !== inSize) {
            throw new Error(`Input matrix has different size. Expected size ${inSize}, got ${input.length}`);
        }
    }

    protected _assertInputSize2d(input: matrix.Matrix2D) {
        if (!input || !input[0]) {
            throw new Error("Input data missing or in wrong format");
        }

        const inSize = this.layers[0].size;
        if (input[0].length !== inSize) {
            throw new Error(`Input matrix has different size. Expected size ${inSize}, got ${input[0].length}`);
        }
    }

    protected _assertExpectedSize(expected: matrix.Matrix1D) {
        if (!expected) {
            throw new Error("Expected data missing or in wrong format");
        }

        const outSize = this.layers[this.layers.length - 1].size;
        if (expected.length !== outSize) {
            throw new Error(`Expected matrix has different size. Expected size ${outSize}, got ${expected.length}`);
        }
    }

    protected _assertExpectedSize2d(expected: matrix.Matrix2D) {
        if (!expected || !expected[0]) {
            throw new Error("Expected data missing or in wrong format");
        }

        const outSize = this.layers[this.layers.length - 1].size;
        if (expected[0].length !== outSize) {
            throw new Error(`Expected matrix has different size. Expected size ${outSize}, got ${expected[0].length}`);
        }
    }

    protected _assertInputOutputSize(input: matrix.Matrix2D, expected: matrix.Matrix2D) {
        if (input.length !== expected.length) {
            throw new Error(`Inconsistent data length: input length ${input.length} != expected length ${expected.length}`)
        }
    }
}