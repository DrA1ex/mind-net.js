interface IWorker {
    postMessage(message: any, transferList?: Transferable[]): void;

    on(eventName: string, handler: (...args: any[]) => void): void;
    off(eventName: string, handler: (...args: any[]) => void): void;
    once(eventName: string, handler: (...args: any[]) => void): void;

    terminate(): Promise<void>;
}

type HandlerType = (event: any) => void;

class BrowserWorker implements IWorker {
    private _handlers = new Map<HandlerType, HandlerType>();

    constructor(public readonly worker: Worker) {}

    on(eventName: string, handler: HandlerType): void {
        function _handler(e: any) {
            handler(e.data);
        }

        this._handlers.set(handler, _handler);
        this.worker.addEventListener(eventName, _handler);
    }

    off(eventName: string, handler: HandlerType): void {
        const _handler = this._handlers.get(handler);
        this.worker.removeEventListener(eventName, _handler ?? handler);
    }

    once(eventName: string, handler: (...args: any[]) => void): void {
        const _handler = (e: any) => {
            this.off(eventName, _handler);
            handler(e.data);
        }

        this.on(eventName, handler);
    }
    postMessage(message: any, transferList?: Transferable[]): void {
        this.worker.postMessage(message, transferList as any);
    }

    async terminate(): Promise<void> {
        this.worker.terminate();
    }
}

class NodeWorker {
    constructor(public readonly worker: any) {}

    on(eventName: string, handler: HandlerType): void {
        this.worker.on(eventName, handler);
    }

    off(eventName: string, handler: HandlerType): void {
        this.worker.off(eventName, handler);
    }

    once(eventName: string, handler: HandlerType): void {
        this.worker.once(eventName, handler);
    }
    postMessage(message: any, transferList?: Transferable[]): void {
        this.worker.postMessage(message, transferList as any);
    }

    async terminate(): Promise<void> {
        await this.worker.terminate();
    }
}

type WorkerT = new (...args: any) => IWorker;

async function getWorkerT() {
    let wrapperT: WorkerT;
    let workerT: any;
    if (typeof Worker !== "undefined") {
        wrapperT = BrowserWorker
        workerT = Worker;
    }
    // @ts-ignore
    else if (typeof window === "undefined") {
        const path = "worker_threads";
        const WorkerThreads = await import(/* webpackIgnore: true */ path);
        wrapperT = NodeWorker;
        workerT = WorkerThreads.Worker;
    } else {
        throw new Error("Unsupported environment: Worker doesn't exists");
    }

    return [wrapperT, workerT];
}

export {getWorkerT, IWorker, WorkerT};