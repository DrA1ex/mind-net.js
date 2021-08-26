import * as matrix from "./matrix";
import * as iter from "./iter";

import {ILayer, IOptimizer} from "./base";
import {Optimizers, OptimizerT} from "./optimizers";
import {GlobalPool, MemorySlice} from "./memory";

export type NeuralNetworkSnapshot = { weights: matrix.Matrix2D[], biases: matrix.Matrix1D[] };

export class SequentialModel {
    private compiled: boolean = false;
    private epoch: number = 0;

    readonly layers: ILayer[] = [];
    readonly optimizer: IOptimizer;

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
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];

            const prevSize = i > 0 ? this.layers[i - 1].size : 0;
            layer.build(i, prevSize);
        }

        this.compiled = true;
    }

    compute(input: matrix.Matrix1D): matrix.Matrix1D {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }

        let result = GlobalPool.allocFrom(input);
        for (let i = 1; i < this.layers.length; i++) {
            const layer = this.layers[i];

            const newValue = layer.step(result);
            result.free();

            result = layer.activation.value(newValue);
            newValue.free();
        }

        const unmanagedResult = MemorySlice.from(result.data);
        result.free();

        return unmanagedResult;
    }

    train(input: matrix.Matrix1D, expected: matrix.Matrix1D) {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }

        const activations: matrix.Matrix1D[] = new Array(this.layers.length);
        const primes: matrix.Matrix1D[] = new Array(this.layers.length);
        activations[0] = input;
        primes[0] = input;
        for (let i = 1; i < this.layers.length; i++) {
            primes[i] = this.layers[i].step(activations[i - 1]);
            activations[i] = this.layers[i].activation.value(primes[i]);
        }

        let errors = matrix.sub(expected, activations[activations.length - 1]);
        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];

            const change = this.optimizer.step(layer, primes[i], errors, this.epoch);
            for (let j = 0; j < layer.size; j++) {
                matrix.matrix1d_binary_in_place_op(layer.weights[j], activations[i - 1], (w, a) => w + a * change.data[j]);
            }
            matrix.matrix1d_binary_in_place_op(layer.biases, change, (b, c) => b + c);

            change.free()
            if (i > 1) {
                const newErrors = matrix.dot_2d_translated(layer.weights, errors);
                errors.free();
                errors = newErrors;
            }
        }

        errors.free();

        for (const item of iter.chain(activations.slice(1), primes.slice(1))) {
            item.free();
        }

        this.epoch += 1;
    }

    getSnapshot(): NeuralNetworkSnapshot {
        return {
            weights: this.layers.slice(1).map(l => l.weights.map(w => MemorySlice.from(w.data))),
            biases: this.layers.slice(1).map(l => MemorySlice.from(l.biases.data))
        };
    }
}

export const Models = {
    Sequential: SequentialModel
}