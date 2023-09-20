import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import {spawn} from "threads"

import {
    FileAsyncReader,
    ObservableStreamLoader,
    CommonUtils, ProgressUtils,
} from "../../neural-network/neural-network";

import {ModelParams, WorkerT} from "../../workers/demo3/demo3.worker";
import {BinaryImageDrawerComponent} from "../../components/binary-image-drawer/binary-image-drawer.component";
import * as FileInteraction from "../../utils/file-interaction";

@Component({
    selector: 'app-demo3',
    templateUrl: './demo3.component.html',
    styleUrls: ['./demo3.component.css']
})
export class Demo3Component implements AfterViewInit {
    @ViewChild('drawingCanvas', {static: true})
    drawingCanvasRef!: ElementRef<HTMLCanvasElement>;

    @ViewChild("generatedImage")
    generatedImage!: BinaryImageDrawerComponent;

    private drawingContext!: CanvasRenderingContext2D;
    private modelParams?: ModelParams;

    public isDrawing = false;
    public isRendering = false;

    private renderingRequested = false;

    worker!: WorkerT;

    brushColor!: string;
    brushSize!: number

    modelDetails: string = "Load model to view details";
    progress = {
        loaded: 0,
        total: 0,
        progressFn: (loaded: number, total: number) => {
            this.progress.loaded = loaded;
            this.progress.total = total;
        },

        reset: () => {
            this.progress.loaded = 0;
            this.progress.total = 0;
        },
        converter: CommonUtils.formatByteSize
    };

    ngAfterViewInit() {
        this.drawingContext = this.drawingCanvasRef.nativeElement.getContext('2d')!;
        this.resetDrawing();
    }

    startDrawing(event: MouseEvent | TouchEvent) {
        this.isDrawing = true;
        this.drawingContext.beginPath();
        this.draw(event);
    }

    draw(event: MouseEvent | TouchEvent) {
        if (!this.isDrawing) return;
        if (event instanceof MouseEvent && event.buttons !== 1) return this.stopDrawing();

        const canvas = this.drawingCanvasRef.nativeElement;
        const rect = canvas.getBoundingClientRect();

        const clientX = event instanceof MouseEvent ? event.clientX : event.touches[0].clientX;
        const clientY = event instanceof MouseEvent ? event.clientY : event.touches[0].clientY;

        const x = (clientX - rect.left) / rect.width;
        const y = (clientY - rect.top) / rect.height;

        this.drawingContext.lineWidth = this.brushSize;
        this.drawingContext.strokeStyle = this.brushColor;

        this.drawingContext.lineTo(canvas.width * x, canvas.height * y);
        this.drawingContext.stroke();

        event.preventDefault();
    }

    stopDrawing() {
        if (!this.isDrawing) return;

        this.isDrawing = false;
        this.updateModelPrediction();
    }

    resetDrawing() {
        this.drawingContext.fillStyle = "white";
        this.drawingContext.fillRect(0, 0, this.drawingCanvasRef.nativeElement.width, this.drawingCanvasRef.nativeElement.height);
    }

    async loadImage() {
        try {
            const file = await FileInteraction.openFile('image/*', false) as File;
            if (!file) return;

            const reader = new FileAsyncReader(file);
            const loader = new ObservableStreamLoader(reader, this.progress.progressFn);

            const data = await loader.load();
            await this.loadImageFromArrayBuffer(data, file.type);

            this.updateModelPrediction();
        } finally {
            this.progress.reset();
        }
    }

    async loadModel() {
        if (!this.worker) {
            this.worker = await spawn<WorkerT>(
                new Worker(new URL('../../workers/demo3/demo3.worker', import.meta.url))
            );
        }

        const progressSub = this.worker.progress()
            .subscribe((data: any) => this.progress.progressFn(data.current, data.total));

        try {
            const files = await FileInteraction.openFile('application/json,.bin', true) as File[];
            if (!files?.length) return;

            const res = await this.worker.loadModel(files);
            this.modelParams = res;
            this.modelDetails = res.description;

            this.updateModelPrediction();
        } catch (err) {
            if (err instanceof Error) {
                this.modelDetails = `Error: ${err.message}`;
            } else {
                this.modelDetails = `Unknown error`;
            }
        } finally {
            this.progress.reset();
            progressSub.unsubscribe();
        }
    }

    updateModelPrediction() {
        if (!this.modelParams) return;

        if (!this.isRendering) {
            this.isRendering = true;
            this._updateModelPredictionImpl().finally(() => {
                this.isRendering = false;

                if (this.renderingRequested) {
                    this.renderingRequested = false;
                    this.updateModelPrediction();
                }
            });
        } else if (!this.renderingRequested) {
            this.renderingRequested = true;
        }
    }

    async _updateModelPredictionImpl() {
        const size = Math.sqrt(this.modelParams!.inputSize);
        const data = this.getImageDataFromCanvas(this.drawingCanvasRef.nativeElement, size, size);

        const result = await this.worker.compute(data);
        this.generatedImage.draw(result.buffer, result.size, result.size);
    }

    getImageDataFromCanvas(canvas: HTMLCanvasElement, targetWidth: number, targetHeight: number) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = targetWidth;
        tempCanvas.height = targetHeight;

        const tempContext = tempCanvas.getContext('2d')!;
        tempContext.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, targetWidth, targetHeight);

        return tempContext.getImageData(0, 0, targetWidth, targetHeight).data;
    }

    loadImageFromArrayBuffer(arrayBuffer: ArrayBuffer, mimeType: string) {
        const canvas = this.drawingCanvasRef.nativeElement;

        const blob = new Blob([arrayBuffer], {type: mimeType});
        const url = URL.createObjectURL(blob);

        const image = new Image();
        return new Promise<void>((resolve, reject) => {
            image.onload = () => {
                this.drawingContext.drawImage(image, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(url);
                resolve();
            };

            image.onerror = reject;
            image.src = url;
        });
    }
    protected readonly ProgressUtils = ProgressUtils;
}