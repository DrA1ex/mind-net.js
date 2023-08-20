import {Component, ViewChild} from '@angular/core';

import JSZip from "jszip";
import {BinaryImageDrawerComponent} from "../../components/binary-image-drawer/binary-image-drawer.component";
import {DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_NN_PARAMS} from "../../workers/demo2/gan.worker.consts";

import * as fileInteraction from "../../utils/file-interaction";
import * as image from "../../utils/image";
import * as matrix from "../../neural-network/engine/matrix";
import {HttpClient} from "@angular/common/http";
import {AsyncSubject} from "rxjs";

@Component({
    selector: 'app-demo2',
    templateUrl: './demo2.component.html',
    styleUrls: ['./demo2.component.css']
})
export class Demo2Component {
    @ViewChild("generatedImage")
    generatedImage!: BinaryImageDrawerComponent;
    @ViewChild("trainingImage")
    trainingImage!: BinaryImageDrawerComponent;

    private nnWorker!: Worker;
    private modelRequest?: AsyncSubject<any>;
    modelLoading = false;

    nnInputSize: number = DEFAULT_NN_PARAMS[0];
    nnHiddenLayers: string = DEFAULT_NN_PARAMS[1].join(" ");
    nnGenOutSize: number = DEFAULT_NN_PARAMS[2];
    learningRate: number = DEFAULT_LEARNING_RATE;
    batchSize: number = DEFAULT_BATCH_SIZE;

    currentIteration: number = 0;
    currentBatch: number = 0;
    totalBatches: number = 0;
    speed: number = 0;

    fileLoading = false;
    fileProcessingCurrent: number = 0;
    fileProcessingTotal: number = 0;

    constructor(private http: HttpClient) {
        this.nnWorker = new Worker(new URL('../../workers/demo2/gan.worker', import.meta.url));

        this.nnWorker.onmessage = ({data}) => {
            switch (data.type) {
                case "training_data":
                    this.generatedImage.draw(data.generatedData, data.width, data.height);
                    this.trainingImage.draw(data.trainingData, data.width, data.height);
                    break;

                case "progress":
                    this.currentIteration = data.epoch ?? this.currentIteration;
                    this.currentBatch = data.batchNo ?? this.currentBatch;
                    this.totalBatches = data.batchCount ?? this.totalBatches;
                    this.speed = data.speed ?? this.speed;
                    this.nnGenOutSize = data.nnParams && data.nnParams[2] || this.nnGenOutSize;
                    break;

                case "model_dump":
                    if (!this.modelRequest) {
                        console.error("Model dump not expected");
                        break;
                    }

                    this.modelRequest.next(data.dump);
                    this.modelRequest.complete();
                    this.modelRequest = undefined;
                    break
            }
        }
    }

    refresh() {
        const config = [
            +this.nnInputSize,
            this.nnHiddenLayers.split(" ").map(v => Number.parseInt(v)),
            +this.nnGenOutSize
        ];

        this.nnWorker.postMessage({
            type: "refresh",
            learningRate: +this.learningRate,
            batchSize: +this.batchSize,
            layers: config,
        });
    }

    async loadData() {
        if (this.fileLoading) {
            return;
        }

        try {
            const file = await fileInteraction.openFile('application/zip', false) as File;
            if (!file) {
                return;
            }

            this.fileLoading = true;

            const zip = new JSZip();
            const loaded = await zip.loadAsync(file);

            const items = Object.values(loaded.files)
                .map(f => ({match: f.name.match(/.*.png$/), file: f}))
                .filter(p => p.match);

            const result: matrix.Matrix1D[] = new Array();
            this.fileProcessingTotal = items.length;
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                try {
                    const imgData = await image.getTrainingDataFromImage(await item.file.async("arraybuffer"))
                    result.push(imgData);
                } catch (e) {
                    console.warn(`Unable to load file: ${item.match}`)
                }

                if (i % 30 === 0) {
                    this.fileProcessingCurrent = i;
                }
            }

            this.nnWorker.postMessage({type: "set_data", data: result});
        } finally {
            this.fileLoading = false;
        }
    }

    async loadPredefined(name: string) {
        if (this.fileLoading) {
            return;
        }

        try {
            this.fileLoading = true;

            const data = await this.http.get(`./assets/dataset/${name}`).toPromise();
            this.nnWorker.postMessage({type: "set_data", data});
        } finally {
            this.fileLoading = false;
        }
    }

    async saveModel() {
        this.modelRequest = new AsyncSubject<any>();
        this.nnWorker.postMessage({type: "dump"});

        this.modelLoading = true;
        try {
            const data = await this.modelRequest.toPromise();
            fileInteraction.saveFile(JSON.stringify(data), 'dump.json', 'application/json');
        } finally {
            this.modelLoading = false;
        }
    }

    async loadModel() {
        const file = await fileInteraction.openFile('application/json', false) as File;
        if (!file) {
            return;
        }

        const content = await file.text();
        const dump = JSON.parse(content);

        this.nnInputSize = dump.generator.layers[0].size;
        this.nnHiddenLayers = dump.generator.layers.slice(1, dump.generator.layers.length - 1).map((l: any) => l.size).join(" ");
        this.nnGenOutSize = dump.generator.layers[dump.generator.layers.length - 1].size;
        this.learningRate = dump.optimizer.params.lr;

        this.nnWorker.postMessage({type: "load_dump", dump});
    }
}