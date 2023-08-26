import {one, zero, zero_2d} from "../matrix";

import {ILayer, IModel} from "../base";
import {ModelBase} from "./base";

export class ChainModel extends ModelBase {
    layers: ILayer[] = [];
    private trainable: boolean[] = [];
    private modelByLayer = new Map<ILayer, [IModel, boolean]>();

    readonly models: IModel[] = [];

    addModel(model: IModel, trainable = true): this {
        this.compiled = false;

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
                this.cache.set(layer, {
                    deltaWeights: zero_2d(layer.size, layer.prevSize),
                    deltaBiases: zero(layer.size),
                    mask: one(layer.size),
                });
            }

            this.layers.push(...layers);
        }

        this.compiled = true;
    }

    protected _applyLayerDelta(layer: ILayer, batchSize: number) {
        const [, trainable] = this.modelByLayer.get(layer)!;
        if (!trainable) return;

        super._applyLayerDelta(layer, batchSize);
    }
}