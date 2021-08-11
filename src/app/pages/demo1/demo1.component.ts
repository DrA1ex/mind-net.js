import {Component, ViewChild} from '@angular/core';
import {DEFAULT_LEARNING_RATE, DEFAULT_NN_LAYERS, MAX_TRAINING_ITERATION, Point} from "../../workers/demo1/nn.worker.consts";

import * as fileInteraction from '../../utils/file-interaction';
import {PlotDrawerComponent} from "../../components/plot-drawer/plot-drawer.component";
import {NeuralNetworkDrawerComponent} from "../../components/neural-network-drawer/neural-network-drawer.component";


@Component({
    selector: 'demo1-page',
    templateUrl: './demo1.component.html',
    styleUrls: ['./demo1.component.css']
})
export class Demo1Component {
    @ViewChild('plotDrawer')
    plotDrawer!: PlotDrawerComponent;

    @ViewChild('nnDrawer')
    nnDrawer!: NeuralNetworkDrawerComponent;

    private nnWorker!: Worker;

    defaultPointType: number = 0;

    layersConfig: string = DEFAULT_NN_LAYERS.join(" ");
    learningRate: number = DEFAULT_LEARNING_RATE;
    maxIterations = MAX_TRAINING_ITERATION;

    currentIteration: number = 0;
    training: boolean = false;
    points: any[] = [];

    constructor() {
        this.nnWorker = new Worker(new URL('../../workers/demo1/nn.worker', import.meta.url));

        this.nnWorker.onmessage = ({data}) => {
            switch (data.type) {
                case "training_data":
                    console.log(`*** Transfer took ${performance.now() - data.t}ms`);

                    this.currentIteration = data.iteration;
                    this.training = this.currentIteration < this.maxIterations && this.points.length > 0;

                    this.plotDrawer.drawSnapshot(data.state, data.width, data.height);
                    this.nnDrawer.drawSnapshot(data.nnSnapshot);
                    break;
            }
        }
    }


    handleMouseEvent($event: MouseEvent) {
        const element = $event.target as Element;
        const rect = element.getBoundingClientRect();
        const x = $event.clientX - rect.left, y = $event.clientY - rect.top;

        const pointType = +($event.button > 0 || $event.altKey) || this.defaultPointType;
        const point = {
            x: x / element.clientWidth,
            y: y / element.clientHeight,
            type: pointType
        }

        this.plotDrawer.addPoint(point);
        this.points.push(point);
        this.nnWorker.postMessage({type: 'add_point', point: point});
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
        this.plotDrawer.clearPoints();

        for (const point of this.points) {
            this.plotDrawer.addPoint(point);
        }

        this.training = this.currentIteration < this.maxIterations && this.points.length > 0;
        this.nnWorker.postMessage({type: "set_points", points: this.points});
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
