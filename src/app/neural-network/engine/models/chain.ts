import * as matrix from "../matrix"

import {SequentialModel} from "./sequential";
import {ILayer} from "../base";

export class ChainModel {
    private compiled: boolean = false;
    private epoch: number = 0;

    private trainable: boolean[] = [];
    private layers: ILayer[] = [];
    private cache = new Map<ILayer, matrix.Matrix1D>();
    private modelByLayer = new Map<ILayer, [SequentialModel, boolean]>();

    readonly models: SequentialModel[] = [];

    addModel(model: SequentialModel, trainable = true): this {
        this.models.push(model);
        this.trainable.push(trainable);
        return this;
    }

    compile() {
        if (this.compiled) {
            return;
        }

        for (let i = 0; i < this.models.length; i++) {
            const model = this.models[i];
            if (i > 0) {
                const prevModel = this.models[i - 1];

                const currentSize = model.layers[0].size
                const prevSize = prevModel.layers[prevModel.layers.length - 1].size;

                if (currentSize !== prevSize) {
                    throw new Error(`Models in chain has different in-out sizes: ${prevSize} != ${currentSize}`);
                }
            }

            model.compile();

            let layers
            if (i > 0) {
                layers = model.layers.slice(1);
            } else {
                layers = model.layers;
            }

            for (const layer of layers) {
                this.modelByLayer.set(layer, [model, this.trainable[i]]);
                this.cache.set(layer, new Array(layer.size));
            }

            this.layers.push(...layers);
        }

        this.compiled = true;
    }

    compute(input: matrix.Matrix1D): matrix.Matrix1D {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
        }

        let a = input;
        for (const model of this.models) {
            a = model.compute(a);
        }

        return a;
    }

    train(input: matrix.Matrix1D, expected: matrix.Matrix1D) {
        if (!this.compiled) {
            throw new Error("Model should be compiled before usage");
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
            const [model, trainable] = this.modelByLayer.get(layer)!;

            const change = model.optimizer.step(layer, primes[i], errors, this.epoch);

            if (trainable) {
                for (let j = 0; j < layer.size; j++) {
                    matrix.matrix1d_binary_in_place_op(layer.weights[j], activations[i - 1], (w, a) => w + a * change[j]);
                }
                matrix.matrix1d_binary_in_place_op(layer.biases, change, (b, c) => b + c);

                if (i > 1) {
                    errors = matrix.dot_2d_translated(layer.weights, errors);
                }
            } else {
                const tmpWeights = new Array(layer.size);
                for (let j = 0; j < layer.size; j++) {
                    tmpWeights[j] = matrix.matrix1d_binary_op(layer.weights[j], activations[i - 1], (w, a) => w + a * change[j]);
                }

                errors = matrix.dot_2d_translated(tmpWeights, errors);
            }
        }

        this.epoch += 1;
    }
}