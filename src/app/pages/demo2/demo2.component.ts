import {Component, ViewChild} from '@angular/core';

import * as JSZip from "jszip";
import {BinaryImageDrawerComponent} from "../../components/binary-image-drawer/binary-image-drawer.component";
import {DEFAULT_LEARNING_RATE, DEFAULT_NN_PARAMS} from "../../workers/demo2/gan.worker.consts";

import * as fileInteraction from "../../utils/file-interaction";
import * as image from "../../utils/image";
import * as matrix from "../../neural-network/engine/matrix";

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

    nnInputSize: number = DEFAULT_NN_PARAMS[0];
    nnGenLayers: string = DEFAULT_NN_PARAMS[1].join(" ");
    nnGenOutSize: number = DEFAULT_NN_PARAMS[2]
    nnDisLayers: string = DEFAULT_NN_PARAMS[3].join(" ");
    learningRate: number = DEFAULT_LEARNING_RATE;

    currentIteration: number = 0;

    fileLoading = false;
    fileProcessingCurrent: number = 0;
    fileProcessingTotal: number = 0;

    constructor() {
        this.nnWorker = new Worker(new URL('../../workers/demo2/gan.worker', import.meta.url));

        this.nnWorker.onmessage = ({data}) => {
            switch (data.type) {
                case "training_data":
                    this.generatedImage.draw(data.generatedData, data.width, data.height);
                    this.trainingImage.draw(data.trainingData, data.width, data.height);

                    this.currentIteration = data.currentIteration;
                    break;
            }
        }
    }

    refresh() {
        const config = [
            this.nnInputSize, this.nnGenLayers.split(" ").map(v => Number.parseInt(v)),
            this.nnGenOutSize, this.nnDisLayers.split(" ").map(v => Number.parseInt(v))
        ];

        this.nnWorker.postMessage({type: "refresh", learningRate: this.learningRate, layers: config});
    }

    async loadData() {
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

            const result = new Array(items.length);
            this.fileProcessingTotal = items.length;
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                result[i] = await image.getTrainingDataFromImage(await item.file.async("arraybuffer"))

                if (i % 30 === 0) {
                    this.fileProcessingCurrent = i;
                }
            }

            this.nnWorker.postMessage({type: "set_data", data: result});
        } finally {
            this.fileLoading = false;
        }
    }
}