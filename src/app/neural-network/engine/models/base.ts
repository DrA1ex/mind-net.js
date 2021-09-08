import * as iter from "../iter";
import * as matrix from "../matrix";

import {ILayer, IOptimizer} from "../base";
import {buildOptimizer, OptimizerT} from "../optimizers";

export type NeuralNetworkSnapshot = { weights: matrix.Matrix2D[], biases: matrix.Matrix1D[] };

export type BackpropData = { activations: matrix.Matrix1D[], primes: matrix.Matrix1D[] };
export type LayerCache = { activation: matrix.Matrix1D, deltaBiases: matrix.Matrix1D, deltaWeights: matrix.Matrix2D };

export abstract class ModelBase {
    protected _epoch: number = 0;

    protected compiled: boolean = false;
    protected cache = new Map<ILayer, LayerCache>();
    protected readonly optimizer: IOptimizer;

    public get epoch() {
        return this._epoch;
    }

    abstract readonly layers: ILayer[];

    abstract compile(...args: any[]): void

    constructor(optimizer: OptimizerT | IOptimizer = 'sgd') {
        this.optimizer = buildOptimizer(optimizer);
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
            result = matrix.matrix1d_unary_op(layer.step(result), x => layer.activation.value(x), this.cache.get(layer)?.activation);
        }

        return result;
    }

    train(input: matrix.Matrix1D[], expected: matrix.Matrix1D[], batchSize: number = 32) {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }

        const shuffledTrainSet = iter.shuffled(Array.from(iter.zip(input, expected)));
        for (const batch of iter.partition(shuffledTrainSet, batchSize)) {
            this.trainBatch(batch);
        }

        this._epoch += 1;
    }

    public trainBatch(batch: Iterable<[matrix.Matrix1D, matrix.Matrix1D]>) {
        this._clearDelta();

        let count = 0;
        for (const [trainInput, trainExpected] of batch) {
            const data = this._calculateBackpropData(trainInput);
            const loss = this._calculateLoss(data.activations[data.activations.length - 1], trainExpected);
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
            activations[i] = matrix.matrix1d_unary_op(primes[i], x => layer.activation.value(x), this.cache.get(layer)?.activation);
        }

        return {activations, primes};
    }

    protected _calculateLoss(predicted: matrix.Matrix1D, expected: matrix.Matrix1D): matrix.Matrix1D {
        if (expected.length !== predicted.length) {
            throw new Error(`Output matrix has different size. Expected size ${expected.length}, got ${predicted.length}`);
        }

        return matrix.sub(expected, predicted);
    }

    protected _backprop(data: BackpropData, loss: matrix.Matrix1D) {
        const {activations, primes} = data;
        let errors = loss;

        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];
            const {deltaWeights, deltaBiases} = this.cache.get(layer)!;

            const gradient = matrix.matrix1d_unary_op(errors, v => layer.activation.moment(v));

            for (let j = 0; j < layer.size; j++) {
                matrix.matrix1d_binary_in_place_op(deltaWeights[j], activations[i - 1], (w, a) => w + a * gradient[j]);
            }

            matrix.add_to(deltaBiases, errors);

            if (i > 1) {
                errors = matrix.dot_2d_translated(this.layers[i].weights, gradient);
            }
        }
    }

    protected __backprop(data: BackpropData, loss: matrix.Matrix1D) {
        const {activations, primes} = data;
        let errors = loss;

        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];
            const {deltaWeights, deltaBiases} = this.cache.get(layer)!;

            for (let j = 0; j < layer.size; j++) {
                matrix.matrix1d_binary_in_place_op(deltaWeights[j], activations[i - 1], (w, a) => w + a * errors[j]);
            }

            matrix.add_to(deltaBiases, errors);

            if (i > 1) {
                errors = matrix.matrix1d_binary_in_place_op(
                    matrix.dot_2d_translated(this.layers[i].weights, errors), primes[i - 1],
                    (d, z) => d * this.layers[i - 1].activation.moment(z)
                );
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
        this.optimizer.updateWeights(layer, deltaWeights, deltaBiases, batchSize);
    }

    getSnapshot(): NeuralNetworkSnapshot {
        return {
            weights: this.layers.slice(1).map(l => l.weights.map(w => matrix.copy(w))),
            biases: this.layers.slice(1).map(l => matrix.copy(l.biases))
        };
    }
}