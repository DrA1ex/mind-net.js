import * as iter from "../iter";
import * as matrix from "../matrix";

import {ILayer, IOptimizer, ILoss} from "../base";
import {buildLoss, LossT} from "../loss";
import {buildOptimizer, OptimizerT} from "../optimizers";

export type NeuralNetworkSnapshot = { weights: matrix.Matrix2D[], biases: matrix.Matrix1D[] };

export type BackpropData = { activations: matrix.Matrix1D[], primes: matrix.Matrix1D[] };
export type LayerCache = { activation: matrix.Matrix1D, deltaBiases: matrix.Matrix1D, deltaWeights: matrix.Matrix2D };

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
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }

        if (input.length !== this.layers[0].size) {
            throw new Error(`Input matrix has different size. Expected size ${this.layers[0].size}, got ${input.length}`);
        }

        let result = input;
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];
            result = layer.activation.value(layer.step(result), this.cache.get(layer)?.activation);
        }

        return result.concat();
    }

    train(input: matrix.Matrix1D[], expected: matrix.Matrix1D[], batchSize: number = 32) {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }

        this.optimizer.beforePass();
        const shuffledTrainSet = iter.shuffle(Array.from(iter.zip(input, expected)));
        for (const batch of iter.partition(shuffledTrainSet, batchSize)) {
            this.trainBatch(batch);
        }

        this.optimizer.afterPass();
        this._epoch += 1;
    }

    public trainBatch(batch: Iterable<[matrix.Matrix1D, matrix.Matrix1D]>) {
        this._clearDelta();

        let count = 0;
        for (const [trainInput, trainExpected] of batch) {
            const data = this._calculateBackpropData(trainInput);
            const loss = this.loss.calculateError(data.activations[data.activations.length - 1], trainExpected);
            this._backprop(data, loss);
            ++count;
        }

        this._applyDelta(count);
    }

    protected _calculateBackpropData(input: matrix.Matrix1D): BackpropData {
        if (input.length !== this.layers[0].size) {
            throw new Error(`Input matrix has different size. Expected size ${this.layers[0].size}, got ${input.length}`);
        }

        const activations = new Array(this.layers.length);
        const primes = new Array(this.layers.length);
        activations[0] = input;
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];

            primes[i] = layer.step(activations[i - 1]);
            activations[i] = layer.activation.value(primes[i], this.cache.get(layer)?.activation);
        }

        return {activations, primes};
    }

    protected _backprop(data: BackpropData, loss: matrix.Matrix1D) {
        const {activations, primes} = data;
        let errors = loss;

        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];
            const {deltaWeights, deltaBiases} = this.cache.get(layer)!;

            const gradient = this.optimizer.step(layer, activations[i], primes[i], errors, this.epoch);
            for (let j = 0; j < layer.size; j++) {
                matrix.matrix1d_binary_in_place_op(deltaWeights[j], activations[i - 1],
                    (w, a) => w + a * gradient[j]);
            }

            matrix.add_to(deltaBiases, gradient);

            if (i > 1) {
                //TODO: cache output array
                errors = matrix.dot_2d_translated(this.layers[i].weights, gradient);
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
}