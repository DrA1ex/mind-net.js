import {Subject} from "threads/observable";

import {ProgressFn} from "../../neural-network/utils/fetch";
import {ProgressUtils} from "../../neural-network/neural-network";


export class Progress {
    private _current = 0;
    private _total = 0;

    private readonly _throttledSendProgress: ProgressFn;

    readonly subject = new Subject();
    offset = 0;

    get current() {return this._current;}
    set current(value: number) {
        this._current = value;
        this.refresh();
    }

    get total() {return this._total;}
    set total(value: number) {
        this._total = value;
        this.refresh();
    }

    constructor(delay = 100) {
        this._throttledSendProgress = ProgressUtils.throttle(
            (current, total) => this.subject.next({current, total}),
            ProgressUtils.ValueLimit.inclusive,
            delay
        );
    }

    reset() {
        this.offset = 0;
        this._current = 0;
        this._total = 0;

        this.refresh();
    }

    refresh() {
        this._throttledSendProgress(this.offset + this.current, this.total);
    }

    progressFn(current: number, total: number) {
        if (this.total < total) this._total = total;
        else if (this.total < current && this.offset !== 0) this._total = this.offset + total;
        else if (this.total < current) this._total = current;
        this._current = current;

        this.refresh();
    }
}

export type ModelParams = {
    inputSize: number
    outputSize: number
    description: string
}

export enum ModelSourceType {
    file,
    remote,
    stream,
    buffer
}

export type ModelSourceData = {
    source: ModelSourceType
    name: string
    size: number
    data: File | ArrayBuffer | string | ReadableStream
}