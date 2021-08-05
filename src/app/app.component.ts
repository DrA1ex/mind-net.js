import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import {DEFAULT_LEARNING_RATE, DEFAULT_NN_LAYERS, MAX_TRAINING_ITERATION, Point} from "./workers/nn.worker.consts";

import * as fileInteraction from './utils/file-interaction';


@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})
export class AppComponent implements AfterViewInit {
    @ViewChild('canvasBg', {static: false})
    canvasBg!: ElementRef;
    @ViewChild('canvasPoints', {static: false})
    canvasPoints!: ElementRef;

    private nnWorker!: Worker;

    defaultPointType: number = 0;

    layersConfig: string = DEFAULT_NN_LAYERS.join(" ");
    learningRate: number = DEFAULT_LEARNING_RATE;
    maxIterations = MAX_TRAINING_ITERATION;

    currentIteration: number = 0;
    training: boolean = false;
    points: any[] = [];

    constructor() {
        this.nnWorker = new Worker(new URL('./workers/nn.worker', import.meta.url));

        this.nnWorker.onmessage = ({data}) => {
            switch (data.type) {
                case "training_data":
                    console.log(`*** Transfer took ${performance.now() - data.t}ms`);

                    this.currentIteration = data.iteration;
                    this.training = this.currentIteration < this.maxIterations && this.points.length > 0;

                    this.repaint(new Uint8ClampedArray(data.state), data.width, data.height);
                    break;
            }
        }
    }

    ngAfterViewInit(): void {
        const rect = this.canvasBg.nativeElement.getBoundingClientRect();

        this.canvasBg.nativeElement.width = rect.width;
        this.canvasBg.nativeElement.height = rect.height;

        this.canvasPoints.nativeElement.width = rect.width;
        this.canvasPoints.nativeElement.height = rect.height;
    }

    onMouseDown($event: MouseEvent) {
        const canvas = this.canvasPoints.nativeElement as HTMLCanvasElement;
        const rect = canvas.getBoundingClientRect();
        const x = $event.clientX - rect.left, y = $event.clientY - rect.top;

        const pointType = +($event.button > 0 || $event.altKey) || this.defaultPointType;
        const point = {
            x: x / this.canvasBg.nativeElement.clientWidth,
            y: y / this.canvasBg.nativeElement.clientHeight,
            type: pointType
        }

        this.drawPoint(point);
        this.points.push(point);

        this.nnWorker.postMessage({type: 'add_point', point: point});
    }

    drawPoint(point: Point) {
        const canvas = this.canvasPoints.nativeElement as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }

        ctx.lineWidth = 1;
        ctx.strokeStyle = "solid";
        ctx.fillStyle = this.getColorByPointType(point.type);

        const radius = 5;
        const x = point.x * canvas.clientWidth,
            y = point.y * canvas.clientHeight;

        ctx.beginPath()
        ctx.arc(x - radius / 2, y - radius / 2, radius, 0, 2 * Math.PI, false);
        ctx.closePath();
        ctx.fill()
        ctx.stroke()
    }

    refresh() {
        const newLayersConfig = this.layersConfig.split(' ').map(v => Number.parseInt(v)).filter(v => !Number.isNaN(v));
        this.nnWorker.postMessage({
            type: "refresh", config: {
                learningRate: this.learningRate,
                layers: newLayersConfig
            }
        });
    }

    savePoints() {
        fileInteraction.saveFile(JSON.stringify(this.points), 'points.json', 'application/json');
    }

    async loadPoints() {
        const file = await fileInteraction.openFile('application/json', false) as File;
        if (!file) {
            return;
        }

        const content = await file.text();
        this.setPoints(JSON.parse(content));
    }

    removePoints() {
        this.setPoints([]);
    }

    setPoints(points: Point[]) {
        this.points = points;

        const canvas = this.canvasPoints.nativeElement as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);

        for (const point of this.points) {
            this.drawPoint(point);
        }

        this.training = this.currentIteration < this.maxIterations && this.points.length > 0;
        this.nnWorker.postMessage({type: "set_points", points: this.points});
    }

    private repaint(snapshot: Uint8ClampedArray, width: number, height: number) {
        const canvas = this.canvasBg.nativeElement as HTMLCanvasElement;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'low';

        const imageData = new ImageData(snapshot, width, height);
        ctx.putImageData(imageData, 0, 0);
    }

    private getColorByPointType(pointType: number) {
        return `rgb(206, ${Math.min(1, Math.max(0, pointType)) * 255}, 88)`;
    }

    onKeyEvent($event: KeyboardEvent) {
        if ($event.key !== "Alt") {
            return;
        }

        if ($event.type === "keydown") {
            this.defaultPointType = 1;
        } else if ($event.type === "keyup") {
            this.defaultPointType = 0;
        }
    }
}
