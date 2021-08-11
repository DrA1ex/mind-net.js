import {Component, ElementRef, Input, OnChanges, SimpleChanges, ViewChild} from '@angular/core';
import {DelayedChangesProcessor} from "../base/delayed-changes-processor";

@Component({
    selector: 'binary-image-drawer',
    templateUrl: './binary-image-drawer.component.html',
    styleUrls: ['./binary-image-drawer.component.css']
})
export class BinaryImageDrawerComponent implements OnChanges {
    @ViewChild('canvas', {static: false})
    canvas!: ElementRef;

    @Input("canvasWidth")
    canvasWidth: number = 640;
    @Input("canvasHeight")
    canvasHeight: number = 480;

    private refreshHandler = new DelayedChangesProcessor(["canvasWidth", "canvasHeight"], 100, () => this.repaint());
    private data?: { snapshot: Uint8ClampedArray, width: number, height: number };

    ngOnChanges(changes: SimpleChanges): void {
        this.refreshHandler.processChanges(changes);
    }

    public draw(data: ArrayBuffer, width: number, height: number) {
        this.data = {snapshot: new Uint8ClampedArray(data), width, height};
        this.repaint();
    }

    private repaint() {
        if (!this.data) {
            return;
        }

        const {snapshot, width, height} = this.data
        const canvas = this.canvas.nativeElement as HTMLCanvasElement;
        if (canvas.width !== width) {
            canvas.width = width;
        }
        if (canvas.height !== height) {
            canvas.height = height;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }

        const imageData = new ImageData(snapshot, width, height);
        ctx.putImageData(imageData, 0, 0);
    }
}
