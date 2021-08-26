import {Component, ElementRef, Input, SimpleChanges, ViewChild} from '@angular/core';

import {NeuralNetworkSnapshot} from "../../neural-network/engine/models";
import * as iter from "../../neural-network/engine/iter";
import * as matrix from "../../neural-network/engine/matrix";
import {DelayedChangesProcessor} from "../base/delayed-changes-processor";

@Component({
    selector: 'nn-drawer',
    templateUrl: './neural-network-drawer.component.html',
    styleUrls: ['./neural-network-drawer.component.css']
})
export class NeuralNetworkDrawerComponent {
    @ViewChild("canvas", {static: false})
    canvas!: ElementRef;

    @Input("canvasWidth")
    canvasWidth: number = 640;
    @Input("canvasHeight")
    canvasHeight: number = 480;
    @Input("canvasScale")
    canvasScale: number = 4;

    @Input("neuronRadius")
    neuronRadius: number = 10;
    @Input("neuronLineWidth")
    neuronLineWidth: number = 2;
    @Input("neuronColor")
    neuronColorPattern: string = "rgba($value,206,100)";


    @Input("weightMinLineWidth")
    weightMinLineWidth: number = 0;
    @Input("weightMaxLineWidth")
    weightMaxLineWidth: number = 3;

    @Input("weightPositiveColor")
    weightPositiveColor: string = "rgb(101,7,7)";
    @Input("weightNegativeColor")
    weightNegativeColor: string = "rgb(6,52,121)";

    @Input("padding")
    padding: number = 30;

    private refreshHandler = new DelayedChangesProcessor(["canvasWidth", "canvasHeight"], 100, () => this.drawSnapshotImpl());
    private lastSnapshot?: NeuralNetworkSnapshot;

    ngOnChanges(changes: SimpleChanges): void {
        this.refreshHandler.processChanges(changes);
    }

    public drawSnapshot(nnSnapshot: NeuralNetworkSnapshot) {
        this.lastSnapshot = nnSnapshot;
        this.drawSnapshotImpl();
    }

    private drawSnapshotImpl() {
        if (!this.lastSnapshot) {
            return;
        }

        const nnSnapshot = this.lastSnapshot;
        const canvas = this.canvas.nativeElement as HTMLCanvasElement;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const layerCnt = nnSnapshot.biases.length + 1;
        let index = layerCnt - 1;
        for (const weights of iter.reverse(nnSnapshot.weights)) {
            this.drawLayerWeights(ctx, weights, index, layerCnt);
            --index;
        }

        ctx.lineWidth = this.neuronLineWidth;
        ctx.strokeStyle = "black";

        index = layerCnt - 1;
        for (const biases of iter.reverse(nnSnapshot.biases)) {
            this.drawLayerNeurons(ctx, biases, index, layerCnt)
            --index;
        }

        this.drawLayerNeurons(ctx, matrix.zero(nnSnapshot.weights[0][0].length), 0, layerCnt);
    }

    private getNeuronPositionX(layerIndex: number, length: number) {
        return this.canvasScale * (this.padding + layerIndex * ((this.canvasWidth - this.padding * 2) / (length - 1)) + this.neuronRadius / 2);
    }

    private getNeuronPositionY(index: number, length: number) {
        const totalHeight = this.canvasHeight - this.padding * 2;
        const spacesOffset = totalHeight / (2 * length + 1);
        return this.canvasScale * (this.padding + (index * 2 + 1) * spacesOffset + this.neuronRadius / 2 + spacesOffset / 2);
    }

    private drawLayerWeights(ctx: CanvasRenderingContext2D, backWeights: matrix.Matrix2D, layerIndex: number, layerCnt: number) {
        if (layerIndex == 0) {
            return;
        }

        const xPosPrev = this.getNeuronPositionX(layerIndex - 1, layerCnt);
        const xPos = this.getNeuronPositionX(layerIndex, layerCnt);

        for (let i = 0; i < backWeights.length; i++) {
            const neuronBackWeights = backWeights[i];

            const max = matrix.max(neuronBackWeights, true);
            for (let j = 0; j < neuronBackWeights.length; j++) {
                this.drawWeight(ctx, neuronBackWeights.data[j] / max,
                    xPos, this.getNeuronPositionY(i, backWeights.length),
                    xPosPrev, this.getNeuronPositionY(j, neuronBackWeights.length)
                );
            }
        }
    }

    private drawLayerNeurons(ctx: CanvasRenderingContext2D, biases: matrix.Matrix1D, layerIndex: number, layerCnt: number) {
        const xPos = this.getNeuronPositionX(layerIndex, layerCnt);
        const max = matrix.max(biases, true);

        for (let i = 0; i < biases.length; i++) {
            this.drawNeuron(ctx, xPos, this.getNeuronPositionY(i, biases.length), biases.data[i] / max);
        }
    }

    private drawNeuron(ctx: CanvasRenderingContext2D, x: number, y: number, bias: number) {
        const colorValue = Math.min(255, Math.max(0, Math.abs(Math.ceil(255 * bias))));
        ctx.fillStyle = this.neuronColorPattern.replace("$value", colorValue.toString());

        ctx.beginPath();
        ctx.arc(x, y, this.neuronRadius, 0, 2 * Math.PI, false);
        ctx.closePath();

        ctx.fill();
        ctx.stroke();
    }

    private drawWeight(ctx: CanvasRenderingContext2D, weight: number, fromX: number, fromY: number, toX: number, toY: number) {
        ctx.lineWidth = Math.max(this.weightMinLineWidth, this.weightMaxLineWidth * Math.abs(weight));
        ctx.strokeStyle = weight > 0 ? this.weightPositiveColor : this.weightNegativeColor;

        ctx.beginPath()

        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);

        ctx.stroke();
        ctx.closePath();
    }
}
