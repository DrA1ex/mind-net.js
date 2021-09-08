import {ModelBase} from "./base";
import {ILayer} from "../base";
import {zero, zero_2d} from "../matrix";

export class SequentialModel extends ModelBase {
    readonly layers: ILayer[] = [];

    addLayer(layer: ILayer): this {
        this.compiled = false;

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
            this.cache.set(layer, {activation: zero(layer.size), deltaBiases: zero(layer.size), deltaWeights: zero_2d(layer.size, prevSize)});
        }

        this.compiled = true;
    }
}