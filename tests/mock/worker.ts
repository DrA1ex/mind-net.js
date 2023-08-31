import {jest} from '@jest/globals'

import {IWorker, WorkerFactory} from "../../src/app/neural-network/misc/worker";

class MockedWorkerWrapper implements IWorker {
    constructor(public worker: AbstractMockedWorker) {
        worker.init();
    }

    terminate(): Promise<void> {
        return Promise.resolve();
    }
    off(eventName: string, handler: (...args: any[]) => void): void {
        this.worker.off(eventName, handler);
    }
    on(eventName: string, handler: (...args: any[]) => void): void {
        this.worker.on(eventName, handler);
    }
    once(eventName: string, handler: (...args: any[]) => void): void {
        this.worker.once(eventName, handler);
    }
    postMessage(message: any, transferList?: Transferable[]): void {
        this.worker.postMessage(message, transferList);
    }
}

abstract class AbstractMockedWorker {
    abstract ImplT: any;


    private listeners: { [event: string]: Array<(...args: any) => any> } = {};
    private implementation: any;

    init() {
        this.implementation = new this.ImplT();
    }

    off(eventName: string, handler: (...args: any[]) => void): void {
        if (this.listeners[eventName]) {
            const handlers = this.listeners[eventName];
            const index = handlers.indexOf(handler);
            if (index >= 0) {
                handlers.splice(index, 1);
                return;
            }
        }

        throw new Error("Try to unsubscribe, but no subscription found");
    }

    on(eventName: string, handler: (...args: any[]) => void): void {
        if (!this.listeners[eventName]) {
            this.listeners[eventName] = [];
        }

        const handlers = this.listeners[eventName];
        const index = handlers.indexOf(handler);
        if (index >= 0) throw new Error("Try to subscribe, but subscription already exists!");

        handlers.push(handler);
    }

    once(eventName: string, handler: (...args: any[]) => void): void {
        const _handler = (...args: any[]) => {
            this.off(eventName, _handler);
            handler(...args);
        }

        this.on(eventName, _handler);
    }

    postMessage(message: any, transferList?: Transferable[]): void {
        if (!this.listeners["message"] || this.listeners["message"].length === 0) throw new Error("There is no subscribers");

        for (const handler of this.listeners["message"]) {
            const implFn = this.implementation[message.type];
            if (implFn) {
                const args = Object.keys(message.data || {}).map(key => message.data[key])
                const result = implFn.apply(this.implementation, args);
                handler(result);
            } else {
                handler({error: new Error(`Unknown message type ${message.type}`)});
            }
        }
    }
}

export function mockWorker(implT: any) {
    class mockedWorker extends AbstractMockedWorker {
        ImplT = implT;
    }

    let mock: jest.Spied<any>;


    beforeEach(() => {
        mock = jest.spyOn(WorkerFactory.prototype, "getWorkerT");
        mock.mockImplementation(async () =>
            [MockedWorkerWrapper, mockedWorker]
        );
    });

    afterEach(() => {
        mock.mockRestore();
    });
}