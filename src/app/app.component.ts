import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import {MAX_TRAINING_ITERATION, Point, X_STEP, Y_STEP} from "./workers/nn.worker.consts";

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

    layersConfig: string = "5 5";
    learningRate: number = 0.01;
    maxIterations = MAX_TRAINING_ITERATION;

    currentIteration: number = 0;
    training: boolean = false;
    points: any[] = [];

    constructor() {
        this.nnWorker = new Worker(new URL('./workers/nn.worker', import.meta.url));

        this.nnWorker.onmessage = ({data}) => {
            switch (data.type) {
                case "training_data":
                    this.currentIteration = data.iteration;
                    this.training = this.currentIteration < this.maxIterations && this.points.length > 0;

                    this.repaint(data.state);
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

    private repaint(state: [number, number][]) {
        const canvas = this.canvasBg.nativeElement as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }

        const xSteps = Math.ceil(1 / X_STEP),
            ySteps = Math.ceil(1 / Y_STEP);

        const width = canvas.width, height = canvas.height;

        const pointHeight = Y_STEP * height,
            pointWidth = X_STEP * width;

        ctx.clearRect(0, 0, width, height);

        for (let x = 0; x < xSteps; x++) {
            for (let y = 0; y < ySteps; y++) {
                const pointType = state[x * ySteps + y][0];

                const pointColor = this.getColorByPointType(pointType);
                if (pointColor) {
                    ctx.fillStyle = pointColor;
                    ctx.fillRect(x * X_STEP * width, y * Y_STEP * height, pointWidth + 1, pointHeight + 1);
                }
            }
        }
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
