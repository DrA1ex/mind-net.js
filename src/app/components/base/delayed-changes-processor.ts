import {SimpleChanges} from "@angular/core";

export class DelayedChangesProcessor {
    private readonly refreshDelay: number = 100;
    private readonly refreshProperties!: string[];
    private readonly onRefresh: () => void;

    private timerId?: any;

    constructor(props: string[], delay: number, fn: () => void) {
        this.refreshProperties = props;
        this.refreshDelay = delay;
        this.onRefresh = fn;
    }

    public processChanges(changes: SimpleChanges) {
        if (this.refreshProperties.some(v => changes.hasOwnProperty(v))) {
            this.cancelRefresh();

            this.timerId = setTimeout(() => {
                this.timerId = undefined;
                this.onRefresh();
            }, this.refreshDelay);
        }
    }

    public cancelRefresh() {
        if (this.timerId) {
            clearTimeout(this.timerId);
            this.timerId = undefined;
        }
    }
}