import {ILayer, IModel} from "../base";
import {ModelBase} from "./base";

export class ChainModel extends ModelBase {
    layers: ILayer[] = [];
    private layerToModelMap = new Map<ILayer, { model: IModel, index: number }>();

    readonly trainable: boolean[] = [];
    readonly models: IModel[] = [];

    addModel(model: IModel, trainable = true): this {
        if (this.compiled) throw new Error("Adding model to already compiled model is forbidden");

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
                this.layerToModelMap.set(layer, {model, index: i});
                this.layers.push(layer)
            }
        }

        super.compile(true);
    }

    isTrainable(layer: ILayer): boolean {
        const {index} = this.layerToModelMap.get(layer)!;
        return this.trainable[index];
    }
}