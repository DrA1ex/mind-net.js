import * as matrix from "../matrix";
import {ILayer, IOptimizer} from "../base";
import {buildOptimizer, OptimizerT} from "../optimizers";

export type NeuralNetworkSnapshot = { weights: matrix.Matrix2D[], biases: matrix.Matrix1D[] };

type BackpropData = { activations: matrix.Matrix1D[], primes: matrix.Matrix1D[] }

export abstract class ModelBase {
    protected compiled: boolean = false;
    protected epoch: number = 0;
    protected cache = new Map<ILayer, matrix.Matrix1D>();
    protected readonly optimizer: IOptimizer;

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
            result = matrix.matrix1d_unary_op(layer.step(result), x => layer.activation.value(x), this.cache.get(layer));
        }

        return result;
    }

    train(input: matrix.Matrix1D, expected: matrix.Matrix1D) {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }

        const data = this._calculateBackpropData(input)
        const loss = this._calculateLoss(data.activations[data.activations.length - 1], expected);
        this._backprop(data, loss);
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
            activations[i] = matrix.matrix1d_unary_op(primes[i], x => layer.activation.value(x), this.cache.get(layer));
        }

        return {activations, primes};
    }

    protected _calculateLoss(output: matrix.Matrix1D, expected: matrix.Matrix1D): matrix.Matrix1D {
        if (expected.length !== output.length) {
            throw new Error(`Output matrix has different size. Expected size ${output.length}, got ${expected.length}`);
        }

        return matrix.sub(expected, output);
    }

    protected _backprop(data: BackpropData, loss: matrix.Matrix1D) {
        const {activations, primes} = data;
        let errors = loss;

        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];

            const change = this.optimizer.step(layer, primes[i], errors, this.epoch);
            for (let j = 0; j < layer.size; j++) {
                matrix.matrix1d_binary_in_place_op(layer.weights[j], activations[i - 1], (w, a) => w + a * change[j]);
            }

            matrix.matrix1d_binary_in_place_op(layer.biases, change, (b, c) => b + c);

            if (i > 1) {
                errors = matrix.dot_2d_translated(layer.weights, errors);
            }
        }

        this.epoch += 1;
    }

    getSnapshot(): NeuralNetworkSnapshot {
        return {
            weights: this.layers.slice(1).map(l => l.weights.map(w => matrix.copy(w))),
            biases: this.layers.slice(1).map(l => matrix.copy(l.biases))
        };
    }
}