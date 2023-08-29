import {ILayer, IModel} from "../base";
import {LayerCache} from "../models/base";
import {Matrix1D, Matrix2D} from "../matrix";
import {LayerWeights} from "./parallel";
import {Matrix, ParallelUtils, UniversalModelSerializer} from "../../neural-network";

type WorkerSelf = {
    postMessage(message: any): void;
    on(eventName: string, handler: (...args: any) => void): void;
}

async function init() {
    let self: WorkerSelf;
    if (typeof postMessage === "undefined") {
        const url = 'worker_threads'; // Avoid webpack processing
        const WorkerThreads = await import(/* webpackIgnore: true */  url);
        const parentPort = WorkerThreads.parentPort;
        if (!parentPort) {
            throw new Error("File can be used only in Worker");
        }

        self = {
            postMessage: parentPort.postMessage.bind(parentPort),
            on: parentPort.on.bind(parentPort),
        }
    } else {
        self = {
            postMessage: (data) => postMessage(data),
            on: (eventName, handler) => addEventListener(eventName, (e: any) => handler(e.data))
        };
    }

    return self;
}


let Model: IModel;
let Deltas: LayerWeights[];

function initModel(config: string): any {
    Model = UniversalModelSerializer.load(JSON.parse(config));
    (Model as any)["_applyDelta"] = () => {};

    if (!Deltas) {
        Deltas = ParallelUtils.createModelWeights(Model);
    }
}

function syncWeights(weights: LayerWeights[]): any {
    for (let i = 1; i < Model.layers.length; i++) {
        const layer = Model.layers[i];

        Matrix.copy_to_2d(weights[i - 1].weights, layer.weights);
        Matrix.copy_to(weights[i - 1].biases, layer.biases);
    }
}

function trainBatch(batch: [Matrix1D, Matrix1D][]) {
    Model.trainBatch(batch);

    const cache: Map<ILayer, LayerCache> = (Model as any).cache;
    for (let i = 1; i < Model.layers.length; i++) {
        const layer = Model.layers[i];
        const {deltaWeights, deltaBiases} = cache.get(layer)!;

        Matrix.copy_to_2d(deltaWeights, Deltas[i - 1].weights);
        Matrix.copy_to(deltaBiases, Deltas[i - 1].biases);
    }

    return {deltas: Deltas};
}

function beforeTrain(): any {
    Model.beforeTrain();
}

function afterTrain(): any {
    Model.afterTrain();
}

function compute(batch: Matrix2D): { outputs: Float64Array[] } {
    const oSize = Model.outputSize;
    const out = new Float64Array(
        new ParallelUtils.BufferT(batch.length * oSize * Float64Array.BYTES_PER_ELEMENT)
    );

    for (let i = 0; i < batch.length; i++) {
        const output = Model.compute(batch[i]);
        out.set(output, i * oSize);
    }

    return {outputs: ParallelUtils.splitBatches(out, oSize)};
}

init().then((self) => {
    self.on("message", e => {
        let ret;
        switch (e.type) {
            case "initModel":
                ret = initModel(e.data.config);
                break;

            case "syncWeights":
                ret = syncWeights(e.data.weights);
                break;

            case "trainBatch":
                ret = trainBatch(e.data.batch);
                break;

            case "beforeTrain":
                ret = beforeTrain();
                break;

            case "afterTrain":
                ret = afterTrain();
                break;

            case "compute":
                ret = compute(e.data.batch)
                break;

            default:
                ret = {error: new Error(`Unsupported message type '${e.type}'`)};
        }

        self.postMessage(ret);
    });
})