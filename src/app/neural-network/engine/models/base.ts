import * as iter from "../iter";
import * as matrix from "../matrix";

import {ILayer, IOptimizer, ILoss} from "../base";
import {buildLoss, LossT} from "../loss";
import {buildOptimizer, OptimizerT} from "../optimizers";

export type NeuralNetworkSnapshot = { weights: matrix.Matrix2D[], biases: matrix.Matrix1D[] };

export type BackpropData = { activations: matrix.Matrix1D[], primes: matrix.Matrix1D[] };
export type LayerCache = {
    activation: matrix.Matrix1D,
    deltaBiases: matrix.Matrix1D,
    mask: matrix.Matrix1D,
    deltaWeights: matrix.Matrix2D,
    gradientCache: matrix.Matrix1D,
};

export abstract class ModelBase {
    protected _epoch: number = 0;

    protected compiled: boolean = false;
    protected cache = new Map<ILayer, LayerCache>();

    readonly optimizer: IOptimizer;
    readonly loss: ILoss;

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

        let result = input;
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];
            result = layer.activation.value(layer.step(result), this.cache.get(layer)?.activation);
        }

        return result.concat();
    }

    train(input: matrix.Matrix1D[], expected: matrix.Matrix1D[], batchSize: number = 32) {
        this._assertCompiled();
        this._assertInputSize2d(input);
        this._assertExpectedSize2d(expected);

        this.beforeTrain();

        const shuffledTrainSet = iter.shuffle(Array.from(iter.zip(input, expected)));
        for (const batch of iter.partition(shuffledTrainSet, batchSize)) {
            this.trainBatch(batch);
        }

        this.afterTrain();
    }

    public trainBatch(batch: Iterable<[matrix.Matrix1D, matrix.Matrix1D]>) {
        this._assertCompiled();

        this._clearDelta();

        let count = 0;
        for (const [trainInput, trainExpected] of batch) {
            this._assertInputSize(trainInput);
            this._assertExpectedSize(trainExpected);

            const data = this._calculateBackpropData(trainInput);
            const loss = this.loss.calculateError(data.activations[data.activations.length - 1], trainExpected);

            this._backprop(data, loss);
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

    protected _calculateBackpropData(input: matrix.Matrix1D): BackpropData {
        const activations = new Array(this.layers.length);
        const primes = new Array(this.layers.length);
        activations[0] = input;
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const {activation, mask} = this.cache.get(layer)!;
            this._calculateDropoutMask(layer, mask);

            primes[i] = layer.step(activations[i - 1]);
            activations[i] = layer.activation.value(primes[i], activation);

            this._applyDropoutMask(layer, activations[i], mask);
        }

        return {activations, primes};
    }

    protected _calculateDropoutMask(layer: ILayer, dst: matrix.Matrix1D) {
        if (layer.dropout <= 0) return;

        const rate = 1 - layer.dropout;
        const maxZeros = Math.floor(layer.size * (1 - rate));

        let count = 0;
        for (let i = 0; i < layer.size; i++) {
            let v;
            if (count < maxZeros && (v = Math.random() < rate ? 1 : 0) === 0) {
                dst[i] = v;
                count++;
            } else {
                dst[i] = 1 / rate;
            }
        }
    }

    protected _applyDropoutMask(layer: ILayer, values: matrix.Matrix1D, mask: matrix.Matrix1D) {
        if (layer.dropout <= 0) return;

        matrix.mul_to(values, mask);
    }

    protected _backprop(data: BackpropData, loss: matrix.Matrix1D) {
        const {activations, primes} = data;
        let errors = loss;

        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];
            const {deltaWeights, deltaBiases, gradientCache, mask} = this.cache.get(layer)!;

            const gradient = this.optimizer.step(layer, activations[i], primes[i], errors, this.epoch);
            this._applyDropoutMask(layer, gradient, mask);

            for (let j = 0; j < layer.size; j++) {
                matrix.matrix1d_binary_in_place_op(deltaWeights[j], activations[i - 1],
                    (w, a) => w + a * gradient[j]);
            }

            matrix.add_to(deltaBiases, gradient);

            if (i > 1) {
                errors = matrix.dot_2d_translated(this.layers[i].weights, gradient, gradientCache);
            }
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
        const inSize = this.layers[0].size;
        if (input.length !== inSize) {
            throw new Error(`Input matrix has different size. Expected size ${inSize}, got ${input.length}`);
        }
    }

    protected _assertInputSize2d(input: matrix.Matrix2D) {
        const inSize = this.layers[0].size;
        if (input[0].length !== inSize) {
            throw new Error(`Input matrix has different size. Expected size ${inSize}, got ${input[0].length}`);
        }
    }

    protected _assertExpectedSize(expected: matrix.Matrix1D) {
        const outSize = this.layers[this.layers.length - 1].size;
        if (expected.length !== outSize) {
            throw new Error(`Expected matrix has different size. Expected size ${outSize}, got ${expected.length}`);
        }
    }

    protected _assertExpectedSize2d(expected: matrix.Matrix2D) {
        const outSize = this.layers[this.layers.length - 1].size;
        if (expected[0].length !== outSize) {
            throw new Error(`Expected matrix has different size. Expected size ${outSize}, got ${expected[0].length}`);
        }
    }
}