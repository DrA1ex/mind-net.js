import {ParallelWorkerImpl} from "./parallel.worker.impl";

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

init().then((self) => {
    const impl = new ParallelWorkerImpl();

    self.on("message", e => {
        let ret;
        switch (e.type) {
            case "initModel":
                ret = impl.initModel(e.data.config);
                break;

            case "syncWeights":
                ret = impl.syncWeights(e.data.weights);
                break;

            case "trainBatch":
                ret = impl.trainBatch(e.data.batch);
                break;

            case "beforeTrain":
                ret = impl.beforeTrain();
                break;

            case "afterTrain":
                ret = impl.afterTrain();
                break;

            case "compute":
                ret = impl.compute(e.data.batch)
                break;

            default:
                ret = {error: new Error(`Unsupported message type '${e.type}'`)};
        }

        self.postMessage(ret);
    });
})