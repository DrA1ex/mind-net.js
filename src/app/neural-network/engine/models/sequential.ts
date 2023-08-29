import {ModelBase} from "./base";
import {ILayer} from "../base";

export class SequentialModel extends ModelBase {
    readonly layers: ILayer[] = [];

    addLayer(layer: ILayer): this {
        if (this.compiled) throw new Error("Adding layer to already compiled model is forbidden");

        this.layers.push(layer);
        return this;
    }
}