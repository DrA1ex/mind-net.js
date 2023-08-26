import {Iter, Matrix, ModelSerialization} from "../../neural-network";
import {Matrix1D, Matrix2D} from "../matrix";
import {ILayer, IModel} from "../base";
import {getWorkerT, WorkerT, IWorker} from "../../misc/worker";
import {LayerCache} from "../models/base";

export type LayerDelta = {
    weights: Matrix2D;
    biases: Matrix1D;
}

class ModelWorkerWrapper {
    public busy = false;
    private _worker: IWorker;

    constructor(wrapperT: WorkerT, workerT: any, public readonly model: IModel) {
        this._worker = new wrapperT(new workerT(
            new URL("./parallel.worker", import.meta.url),
            {type: "module"}));
    }

    async compute(batch: ArrayLike<number>[]): Promise<ArrayLike<number>[]> {
        const result: any = await this._request("compute", {batch});
        return result.outputs;
    }

    async trainBatch(batch: [ArrayLike<number>, ArrayLike<number>][]): Promise<LayerDelta[]> {
        const result: any = await this._request("trainBatch", {batch});
        return result.deltas;
    }

    async initModel(config: string) {
        return this._request("initModel", {config});
    }

    async afterTrain() {
        return this._request("afterTrain");
    }

    async beforeTrain() {
        return this._request("beforeTrain");
    }

    async syncDeltas(deltas: LayerDelta[], count: number) {
        return this._request("syncDeltas", {deltas, count});
    }

    async terminate() {
        await this._worker.terminate();
    }

    async _request<A, T>(type: string, data?: A): Promise<T> {
        if (this.busy) throw new Error("Worker is busy");
        this.busy = true;

        return new Promise<T>((resolve, reject) => {
            this._worker.once("message", (data: any) => {
                this.busy = false;

                if (data?.error) {
                    reject(data.error);
                } else {
                    resolve(data);
                }
            });

            this._worker.postMessage({type, data});
        });
    }
}

export class ParallelModelWrapper<T extends IModel> {
    static {
        if (typeof SharedArrayBuffer === "undefined") {
            console.warn("The SharedArrayBuffer feature is not available, which may result in a slight to moderate decrease in performance. Learn more: https://mdn.io/SharedArrayBuffer");
        }
    }

    private _initialized: boolean = false;
    private readonly TrainDataCache = new WeakMap<Matrix2D, Float64Array>();

    private readonly _workers: ModelWorkerWrapper[];
    private readonly _deltas: LayerDelta[];

    public readonly parallelism: number;

    constructor(public readonly model: T, parallelism = 4) {
        this.parallelism = Math.max(1, parallelism);

        this._workers = new Array(this.parallelism);

        this._deltas = ParallelUtils.createLayerDeltas(model);
    }

    async init() {
        if (this._initialized) return;

        const [wrapperT, workerT] = await getWorkerT();
        for (let i = 0; i < this.parallelism; i++) {
            this._workers[i] = new ModelWorkerWrapper(wrapperT, workerT, this.model);
        }

        await this._initModel();

        this._initialized = true;
    }

    async train(input: Matrix2D, expected: Matrix2D, {batchSize = 64, cacheTrainData = true} = {}) {
        this._assertInitialized();

        batchSize = Math.max(batchSize, 1);
        if (input.length < batchSize) {
            this.model.train(input, expected, {batchSize});
            return;
        }

        const pInput = this._prepareTrainData(input, cacheTrainData);
        const pOut = this._prepareTrainData(expected, cacheTrainData);

        const trainSet = Array.from(
            Iter.partition(Iter.shuffle(Array.from(Iter.zip(pInput, pOut))), batchSize)
        );

        await this._beforeTrain();

        const count = trainSet.length;
        for (let i = 0; i < count; i += this.parallelism) {
            const tasks: Promise<LayerDelta[]>[] = [];
            let iterCount = 0;
            for (let k = 0; k < this.parallelism && i + k < count; k++) {
                const batch = trainSet[i + k];
                const task = this._workers[k].trainBatch(batch);
                tasks.push(task);
                iterCount += batch.length;
            }

            const trainResults = await Promise.all(tasks);

            this._clearDeltas();
            this._accumulateDelta(trainResults);
            this._applyDeltas(iterCount);

            await this._syncDeltas(iterCount);
        }

        await this._afterTrain();
    }

    async compute(input: Matrix2D, {batchSize = 32, cache = false} = {}) {
        this._assertInitialized();

        if (input.length <= batchSize) {
            return input.map(data => this.model.compute(data));
        }

        const pInput = this._prepareTrainData(input, cache);

        const inputParts = Array.from(Iter.partition(pInput, batchSize));
        const count = inputParts.length;

        const result: ArrayLike<number>[] = [];
        for (let i = 0; i < count; i += this.parallelism) {
            const tasks: Promise<ArrayLike<number>[]>[] = [];
            for (let k = 0; k < this.parallelism && i + k < count; k++) {
                const task = this._workers[k].compute(inputParts[i + k]);
                tasks.push(task);
            }

            const outputs = await Promise.all(tasks)
            result.concat(...outputs);
        }

        return result;
    }

    async terminate() {
        await Promise.all(this._workers.map(w => w.terminate()));
    }

    private async _initModel() {
        const config = JSON.stringify(ModelSerialization.save(this.model));
        await Promise.all(this._workers.map(w => w.initModel(config)));
    }

    private async _beforeTrain() {
        this.model.beforeTrain();
        await Promise.all(this._workers.map(w => w.beforeTrain()));
    }

    private async _afterTrain() {
        this.model.afterTrain();
        await Promise.all(this._workers.map(w => w.afterTrain()));
    }

    private async _syncDeltas(count: number) {
        await Promise.all(this._workers.map(w => w.syncDeltas(this._deltas, count)));
    }

    private _prepareTrainData(data: Matrix2D, cache = true): ArrayLike<number>[] {
        const iSize = data[0].length;
        let fInput = this.TrainDataCache.get(data);
        if (!fInput) {
            fInput = new Float64Array(
                new ParallelUtils.BufferT(data.length * iSize * Float64Array.BYTES_PER_ELEMENT)
            );

            if (cache) this.TrainDataCache.set(data, fInput);
        }

        const pInput = Array.from(
            Iter.map(
                Iter.range(0, data.length),
                i => fInput!.subarray(i * iSize, (i + 1) * iSize)
            )
        );

        Matrix.copy_to_2d(data, pInput as any);

        return pInput;
    }

    private _clearDeltas() {
        for (const delta of this._deltas) {
            delta.weights.forEach(w => w.fill(0));
            delta.biases.fill(0);
        }
    }

    private _accumulateDelta(deltas: LayerDelta[][]) {
        for (const entry of deltas) {
            for (let i = 0; i < entry.length; i++) {
                Matrix.add_to(this._deltas[i].biases, entry[i].biases);

                const wCount = entry[i].weights.length;
                for (let k = 0; k < wCount; k++) {
                    Matrix.add_to(this._deltas[i].weights[k], entry[i].weights[k]);
                }
            }
        }
    }

    private _applyDeltas(batchSize: number) {
        ParallelUtils.updateWeights(this.model, this._deltas, batchSize);
    }

    private _assertInitialized() {
        if (!this._initialized) throw new Error("Wrapper not initialized");
    }
}

export class ParallelUtils {
    static BufferT = typeof SharedArrayBuffer !== "undefined" ? SharedArrayBuffer : ArrayBuffer;

    static createLayerDeltas(model: IModel): LayerDelta[] {
        const result = new Array(model.layers.length - 1);
        for (let i = 1; i < model.layers.length; i++) {
            const layer = model.layers[i];

            const weights = new Float64Array(
                new ParallelUtils.BufferT(layer.size * layer.prevSize * Float64Array.BYTES_PER_ELEMENT)
            );

            const weights2d = Iter.map(
                Iter.range(0, layer.size),
                i => weights.subarray(i * layer.prevSize, (i + 1) * layer.prevSize)
            );

            result[i - 1] = {
                weights: Array.from(weights2d) as any,
                biases: new Float64Array(
                    new ParallelUtils.BufferT(layer.size * Float64Array.BYTES_PER_ELEMENT)
                ) as any
            }
        }

        return result;
    }

    static updateWeights(model: IModel, deltas: LayerDelta[], count: number) {
        const optimizer = model.optimizer;
        const cache: Map<ILayer, LayerCache> = (model as any).cache;

        for (let i = 1; i < model.layers.length; i++) {
            const layer = model.layers[i];
            const {deltaWeights, deltaBiases} = cache.get(layer)!;

            Matrix.copy_to(deltas[i - 1].biases, deltaBiases);
            Matrix.copy_to_2d(deltas[i - 1].weights, deltaWeights);

            optimizer.updateWeights(layer, deltaWeights, deltaBiases, model.epoch, count);
        }
    }
}