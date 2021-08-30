import * as matrix from "../matrix";

import {ILayer, IOptimizer, NeuralNetworkSnapshot} from "../base";
import {Optimizers, OptimizerT} from "../optimizers";

export class SequentialModel {
    private compiled: boolean = false;
    private epoch: number = 0;

    readonly layers: ILayer[] = [];
    readonly optimizer: IOptimizer;

    private cache = new Map<ILayer, matrix.Matrix1D>();

    constructor(optimizer: OptimizerT | IOptimizer = 'sgd',) {
        const optimizer_param = typeof optimizer === "string" ? Optimizers[optimizer] : optimizer
        if (!optimizer_param) {
            throw new Error(`Unknown optimizer type ${optimizer_param}`);
        }

        if (typeof optimizer_param === "object") {
            this.optimizer = optimizer_param;
        } else {
            this.optimizer = new optimizer_param();
        }
    }

    addLayer(layer: ILayer): this {
        this.layers.push(layer);
        return this;
    }

    compile() {
        if (this.compiled) {
            return;
        }

        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];

            const prevSize = i > 0 ? this.layers[i - 1].size : 0;
            layer.build(i, prevSize);
            this.cache.set(layer, new Array(layer.size));
        }

        this.compiled = true;
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

        if (input.length !== this.layers[0].size) {
            throw new Error(`Input matrix has different size. Expected size ${this.layers[0].size}, got ${input.length}`);
        }

        const outSize = this.layers[this.layers.length - 1].size;
        if (expected.length !== outSize) {
            throw new Error(`Output matrix has different size. Expected size ${outSize}, got ${expected.length}`);
        }

        const activations = new Array(this.layers.length);
        const primes = new Array(this.layers.length);
        activations[0] = input;
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];

            primes[i] = layer.step(activations[i - 1]);
            activations[i] = matrix.matrix1d_unary_op(primes[i], x => layer.activation.value(x), this.cache.get(layer));
        }

        let errors = matrix.sub(expected, activations[activations.length - 1]);
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