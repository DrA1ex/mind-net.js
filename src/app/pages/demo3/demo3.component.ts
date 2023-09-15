import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import * as FileInteraction from "../../utils/file-interaction";
import {IModel} from "../../neural-network/engine/base";
import {FileAsyncReader, ObservableStreamLoader} from "../../neural-network/utils/fetch";
import {
    ChainModel,
    CommonUtils,
    ImageUtils,
    ProgressUtils,
    UniversalModelSerializer
} from "../../neural-network/neural-network";
import {BinaryImageDrawerComponent} from "../../components/binary-image-drawer/binary-image-drawer.component";
import * as ColorUtils from "../../neural-network/utils/color";

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
        offset: 0,
        loaded: 0,
        total: 0,
        progressFn: ProgressUtils.throttle(
            (loaded: number, _: number) => {
                this.progress.loaded = this.progress.offset + loaded;
            }, ProgressUtils.ValueLimit.inclusive, 300
        ),
        reset: () => {
            this.progress.offset = 0
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
        if (!this.isDrawing || event.button !== 0) return;

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
            const files = await FileInteraction.openFile('application/json', true) as File[];
            if (!files?.length) return;

            this.progress.total = files.reduce((p, c) => p + c.size, 0);

            const chain = new ChainModel();
            for (const file of files) {
                const reader = new FileAsyncReader(file);
                const loader = new ObservableStreamLoader(reader, this.progress.progressFn);

                const data = await loader.load();
                const config = JSON.parse(new TextDecoder().decode(data));
                const model = UniversalModelSerializer.load(config);
                chain.addModel(model);

                this.progress.offset += file.size;
            }

            this.model = chain;
            this.model.compile();

            const sizes = this.model.layers.map(l => {
                const size = Math.sqrt(l.size);
                if (Number.isInteger(size)) {
                    return `${size}²`;
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

        const input = ColorUtils.transformChannelCount(Array.from(data), 4, 3);
        ColorUtils.transformColorSpace(ColorUtils.rgbToTanh, input, 3, input);

        const output3 = ImageUtils.processMultiChannelData(this.model, input, 3);
        ColorUtils.transformColorSpace(ColorUtils.tanhToRgb, output3, 3, output3);

        const output4 = ColorUtils.transformChannelCount(output3, 3, 4);
        const outData = new Uint8ClampedArray(output4);
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