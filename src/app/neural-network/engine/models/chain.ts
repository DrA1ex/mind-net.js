import * as matrix from "../matrix";
import {zero, zero_2d} from "../matrix";

import {ILayer} from "../base";
import {BackpropData, ModelBase} from "./base";

export class ChainModel extends ModelBase {
    layers: ILayer[] = [];
    private trainable: boolean[] = [];
    private modelByLayer = new Map<ILayer, [ModelBase, boolean]>();

    readonly models: ModelBase[] = [];

    addModel(model: ModelBase, trainable = true): this {
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
                this.cache.set(layer, {activation: zero(layer.size), deltaBiases: zero(layer.size), deltaWeights: zero_2d(layer.size, layer.prevSize)});
            }

            this.layers.push(...layers);
        }

        this.compiled = true;
    }

    protected override _backprop(data: BackpropData, loss: matrix.Matrix1D) {
        const {activations, primes} = data;
        let errors = loss;

        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];
            const [, trainable] = this.modelByLayer.get(layer)!;

            const change = this.optimizer.step(layer, primes[i], errors, this.epoch);

            let layerWeights: matrix.Matrix2D;
            if (trainable) {
                layerWeights = layer.weights;
                matrix.matrix1d_binary_in_place_op(layer.biases, change, (b, c) => b + c);
            } else {
                layerWeights = matrix.copy_2d(layer.weights);
            }

            for (let j = 0; j < layer.size; j++) {
                matrix.matrix1d_binary_in_place_op(layerWeights[j], activations[i - 1], (w, a) => w + a * change[j]);
            }

            if (i > 1) {
                errors = matrix.dot_2d_translated(layerWeights, errors);
            }
        }
    }
}