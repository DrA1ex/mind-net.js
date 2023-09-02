import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import * as FileInteraction from "../../utils/file-interaction";
import {IModel} from "../../neural-network/engine/base";
import {FileAsyncReader, ObservableStreamLoader} from "../../neural-network/utils/fetch";
import {CommonUtils, ImageUtils, UniversalModelSerializer} from "../../neural-network/neural-network";
import {BinaryImageDrawerComponent} from "../../components/binary-image-drawer/binary-image-drawer.component";

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
    private model?: IModel;
    private isDrawing = false;

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

    startDrawing(event: MouseEvent) {
        this.isDrawing = true;
        this.drawingContext.beginPath();
        this.draw(event);
    }

    draw(event: MouseEvent) {
        if (!this.isDrawing) return;
        const canvas = this.drawingCanvasRef.nativeElement;
        const rect = canvas.getBoundingClientRect();

        const x = (event.clientX - rect.left) / rect.width;
        const y = (event.clientY - rect.top) / rect.height;

        this.drawingContext.lineWidth = this.brushSize;
        this.drawingContext.strokeStyle = this.brushColor;

        this.drawingContext.lineTo(canvas.width * x, canvas.height * y);
        this.drawingContext.stroke();
    }

    stopDrawing() {
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
        try {
            const file = await FileInteraction.openFile('application/json', false) as File;
            if (!file) return;

            const reader = new FileAsyncReader(file);
            const loader = new ObservableStreamLoader(reader, this.progress.progressFn);

            const data = await loader.load();
            const config = JSON.parse(new TextDecoder().decode(data));
            this.model = UniversalModelSerializer.load(config);

            const sizes = this.model.layers.map(l => {
                const size = Math.sqrt(l.size);
                if (Number.isInteger(size)) {
                    return `${size}x${size}`;
                }

                return l.size.toString();
            });

            this.modelDetails = `${this.model.constructor.name} (${sizes.join(" -> ")})`;
            this.updateModelPrediction();
        } catch (err) {
            if (err instanceof Error) {
                this.modelDetails = `Error: ${err.message}`;
            } else {
                this.modelDetails = `Unknown error`;
            }
        } finally {
            this.progress.reset();
        }
    }

    updateModelPrediction() {
        if (!this.model) return;

        const size = Math.sqrt(this.model.inputSize);
        const data = this.getImageDataFromCanvas(this.drawingCanvasRef.nativeElement, size, size);

        const input = Array.from(data).map(value => value / 127.5 - 1);
        const output = ImageUtils.processMultiChannelData(this.model, input, 4);

        const mappedOutput = output.map((value, i) => i % 4 !== 3 ? (value + 1) * 127.5 : 255);
        const outData = new Uint8ClampedArray(mappedOutput);
        const outSize = Math.sqrt(this.model.outputSize);

        this.generatedImage.draw(outData.buffer, outSize, outSize);
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
}