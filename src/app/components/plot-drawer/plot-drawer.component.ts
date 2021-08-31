import {Component, ElementRef, Input, OnChanges, Output, SimpleChanges, ViewChild} from '@angular/core';
import {COLOR_A_HEX, COLOR_B_HEX, Point} from "../../workers/demo1/nn.worker.consts";
import * as color from "../../utils/color";

import * as canvasUtils from '../../utils/canvas';
import {BinaryImageDrawerComponent} from "../binary-image-drawer/binary-image-drawer.component";
import {DelayedChangesProcessor} from "../base/delayed-changes-processor";

@Component({
    selector: 'plot-drawer',
    templateUrl: './plot-drawer.component.html',
    styleUrls: ['./plot-drawer.component.css']
})
export class PlotDrawerComponent implements OnChanges {
    @ViewChild('imageDrawer', {static: false})
    imageDrawer!: BinaryImageDrawerComponent;
    @ViewChild('canvasPoints', {static: false})
    canvasPoints!: ElementRef;

    @Input("pointRadius")
    pointRadius: number = 5;
    @Input("lineWidth")
    lineWidth: number = 1;

    @Input("canvasWidth")
    canvasWidth: number = 640;
    @Input("canvasHeight")
    canvasHeight: number = 480;
    @Input("canvasScale")
    canvasScale: number = 2;

    @Output("points")
    points: Point[] = [];

    private refreshHandler = new DelayedChangesProcessor(["canvasWidth", "canvasHeight"], 100, () => {
        this.repaint()
    });

    ngOnChanges(changes: SimpleChanges): void {
        this.refreshHandler.processChanges(changes);
    }

    public drawSnapshot(data: ArrayBuffer, width: number, height: number) {
        this.imageDrawer.draw(data, width, height);
    }

    public addPoint(point: Point) {
        this.points.push(point);
        this.drawPoint(point);
    }

    clearPoints() {
        this.points.splice(0, this.points.length);

        const canvas = this.canvasPoints.nativeElement as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }

    private repaint() {
        const canvas = this.canvasPoints.nativeElement as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);

        for (const point of this.points) {
            this.drawPoint(point);
        }
    }

    private drawPoint(point: Point) {
        const canvas = this.canvasPoints.nativeElement as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }

        const x = (point.x * canvas.clientWidth + this.pointRadius / 2) * this.canvasScale,
            y = (point.y * canvas.clientHeight + this.pointRadius / 2) * this.canvasScale;

        canvasUtils.drawNeuron(ctx, x, y, this.pointRadius * this.canvasScale, this.lineWidth, PlotDrawerComponent.getColorByPointType(point.type));
    }


    private static getColorByPointType(pointType: number) {
        return color.getLinearColorHex(COLOR_A_HEX, COLOR_B_HEX, pointType);
    }
}
