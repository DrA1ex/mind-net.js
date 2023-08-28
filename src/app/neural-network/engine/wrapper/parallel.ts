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

    async compute(batch: ArrayLike<number>[]): Promise<Matrix2D> {
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

export type ParallelWrapperCallOptions = {
    batchSize: number;
    cacheInput: boolean;
    forceUpdateWeights: boolean;
};

export const ParallelWrapperCallOptionsDefaults: ParallelWrapperCallOptions = {
    batchSize: 32,
    cacheInput: true,
    forceUpdateWeights: false,
};

export class ParallelModelWrapper<T extends IModel> {
    static {
        if (typeof SharedArrayBuffer === "undefined") {
            console.warn("The SharedArrayBuffer feature is not available, which may result in a slight to moderate decrease in performance. Learn more: https://mdn.io/SharedArrayBuffer");
        }
    }

    private _initialized: boolean = false;
    private readonly InputDataCache = new WeakMap<Matrix2D, Float64Array>();

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

        await this.syncModel();

        this._initialized = true;
    }

    async train(input: Matrix2D, expected: Matrix2D, options: Partial<ParallelWrapperCallOptions> = {}) {
        this._assertInitialized();

        const opts = {
            ...ParallelWrapperCallOptionsDefaults,
            ...options
        };

        opts.batchSize = Math.max(opts.batchSize, 1);
        if (input.length < opts.batchSize) {
            this.model.train(input, expected, {batchSize: opts.batchSize});
            return;
        }

        if (opts.forceUpdateWeights) {
            await this.syncModel();
        }

        const pInput = this._prepareTrainData(input, opts.cacheInput);
        const pOut = this._prepareTrainData(expected, opts.cacheInput);

        const trainSet = Array.from(
            Iter.partition(Iter.shuffle(Array.from(Iter.zip(pInput, pOut))), opts.batchSize)
        );

        await this.beforeTrain();

        await this.trainBatch(trainSet);

        await this.afterTrain();
    }

    async trainBatch(trainSet: [ArrayLike<number>, ArrayLike<number>][][]) {
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
    }

    async compute(input: Matrix2D, options: Partial<ParallelWrapperCallOptions> = {}) {
        this._assertInitialized();

        const opts = {
            ...ParallelWrapperCallOptionsDefaults,
            ...options
        };

        if (opts.forceUpdateWeights) {
            await this.syncModel();
        }

        if (input.length <= opts.batchSize) {
            return input.map(data => this.model.compute(data));
        }

        const pInput = this._prepareTrainData(input, opts.cacheInput);

        const inputParts = Array.from(Iter.partition(pInput, opts.batchSize));
        const count = inputParts.length;

        const result: Matrix2D = [];
        for (let i = 0; i < count; i += this.parallelism) {
            const tasks: Promise<Matrix2D>[] = [];
            for (let k = 0; k < this.parallelism && i + k < count; k++) {
                const task = this._workers[k].compute(inputParts[i + k]);
                tasks.push(task);
            }

            const outputs = await Promise.all(tasks)
            for (const data of outputs) {
                result.push(...data);
            }
        }

        return result;
    }

    async terminate() {
        await Promise.all(this._workers.map(w => w.terminate()));
        this._initialized = false;
    }

    async syncModel() {
        const config = JSON.stringify(ModelSerialization.save(this.model));
        await Promise.all(this._workers.map(w => w.initModel(config)));
    }

    async beforeTrain() {
        this.model.beforeTrain();
        await Promise.all(this._workers.map(w => w.beforeTrain()));
    }

    async afterTrain() {
        this.model.afterTrain();
        await Promise.all(this._workers.map(w => w.afterTrain()));
    }

    private async _syncDeltas(count: number) {
        await Promise.all(this._workers.map(w => w.syncDeltas(this._deltas, count)));
    }

    private _prepareTrainData(data: Matrix2D, cache = true): ArrayLike<number>[] {
        const iSize = data[0].length;
        let fInput = this.InputDataCache.get(data);
        if (!fInput) {
            fInput = new Float64Array(
                new ParallelUtils.BufferT(data.length * iSize * Float64Array.BYTES_PER_ELEMENT)
            );

            if (cache) this.InputDataCache.set(data, fInput);
        }

        const pInput = ParallelUtils.splitBatches(fInput, iSize);

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
        if (!this._initialized) throw new Error("ParallelModelWrapper can't be used before initialization");
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

            result[i - 1] = {
                weights: ParallelUtils.splitBatches(weights, layer.prevSize),
                biases: new Float64Array(
                    new ParallelUtils.BufferT(layer.size * Float64Array.BYTES_PER_ELEMENT)
                )
            }
        }

        return result;
    }

    static updateWeights(model: IModel, deltas: LayerDelta[], count: number) {
        const optimizer = model.optimizer;
        const cache: Map<ILayer, LayerCache> = (model as any).cache;
        if (!cache) throw new Error("Unsupported model type");

        for (let i = 1; i < model.layers.length; i++) {
            const layer = model.layers[i];
            const {deltaWeights, deltaBiases} = cache.get(layer)!;

            Matrix.copy_to(deltas[i - 1].biases, deltaBiases);
            Matrix.copy_to_2d(deltas[i - 1].weights, deltaWeights);

            optimizer.updateWeights(layer, deltaWeights, deltaBiases, model.epoch, count);
        }
    }

    static splitBatches(data: Float64Array, batchSize: number): Float64Array[] {
        return Matrix.fill(
            i => data.subarray(i * batchSize, (i + 1) * batchSize),
            data.length / batchSize
        );
    }
}