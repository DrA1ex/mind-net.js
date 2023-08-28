import {ModelBase} from "./base";
import {ILayer} from "../base";

export class SequentialModel extends ModelBase {
    readonly layers: ILayer[] = [];

    addLayer(layer: ILayer): this {
        this.compiled = false;

        this.layers.push(layer);
        return this;
    }
}