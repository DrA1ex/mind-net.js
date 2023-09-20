import {ILayer, IModel} from "../base";
import {LayerCache} from "../models/base";
import {Matrix1D, Matrix2D} from "../matrix";
import {LayerWeights} from "./parallel";
import {Matrix, ParallelUtils, UniversalModelSerializer} from "../../neural-network";

export class ParallelWorkerImpl {
    model!: IModel;
    deltas!: LayerWeights[];

    initModel(config: string): any {
        this.model = UniversalModelSerializer.load(JSON.parse(config));
        (this.model as any)["_applyDelta"] = () => {};

        if (!this.deltas) {
            this.deltas = ParallelUtils.createModelWeights(this.model);
        }
    }

    syncWeights(weights: LayerWeights[]): any {
        for (let i = 1; i < this.model.layers.length; i++) {
            const layer = this.model.layers[i];

            Matrix.copy_to_2d(weights[i - 1].weights, layer.weights);
            Matrix.copy_to(weights[i - 1].biases, layer.biases);
        }
    }

    trainBatch(batch: [Matrix1D, Matrix1D][]) {
        this.model.trainBatch(batch);

        const cache: Map<ILayer, LayerCache> = (this.model as any).cache;
        for (let i = 1; i < this.model.layers.length; i++) {
            const layer = this.model.layers[i];
            const {deltaWeights, deltaBiases} = cache.get(layer)!;

            Matrix.copy_to_2d(deltaWeights, this.deltas[i - 1].weights);
            Matrix.copy_to(deltaBiases, this.deltas[i - 1].biases);
        }

        return {deltas: this.deltas};
    }

    beforeTrain(): any {
        this.model.beforeTrain();
    }

    afterTrain(): any {
        this.model.afterTrain();
    }

    compute(batch: Matrix2D): { outputs: Float64Array[] } {
        const oSize = this.model.outputSize;
        const out = new Float64Array(
            new ParallelUtils.BufferT(batch.length * oSize * Float64Array.BYTES_PER_ELEMENT)
        );

        for (let i = 0; i < batch.length; i++) {
            const output = this.model.compute(batch[i]);
            out.set(output, i * oSize);
        }

        return {outputs: Matrix.split_2d(out, oSize)};
    }
}