import {Component, ViewChild} from '@angular/core';

import * as JSZip from "jszip";

import {NeuralNetworkDrawerComponent} from "../../components/neural-network-drawer/neural-network-drawer.component";
import {BinaryImageDrawerComponent} from "../../components/binary-image-drawer/binary-image-drawer.component";
import {DEFAULT_LEARNING_RATE, DEFAULT_NN_LAYERS} from "../../workers/demo1/nn.worker.consts";
import {TrainingData} from "../../workers/demo2/gan.worker.consts";

import * as fileInteraction from "../../utils/file-interaction";
import * as image from "../../utils/image";
import * as matrix from "../../utils/matrix";

@Component({
    selector: 'app-demo2',
    templateUrl: './demo2.component.html',
    styleUrls: ['./demo2.component.css']
})
export class Demo2Component {
    @ViewChild("generatorNn")
    generatorNn!: NeuralNetworkDrawerComponent;
    @ViewChild("discriminatorNn")
    discriminatorNn!: NeuralNetworkDrawerComponent;

    @ViewChild("generatedImage")
    generatedImage!: BinaryImageDrawerComponent;
    @ViewChild("trainingImage")
    trainingImage!: BinaryImageDrawerComponent;

    private nnWorker!: Worker;

    layersConfig: string = DEFAULT_NN_LAYERS.join(" ");
    learningRate: number = DEFAULT_LEARNING_RATE;

    fileLoading = false;

    constructor() {
        this.nnWorker = new Worker(new URL('../../workers/demo2/gan.worker', import.meta.url));

        this.nnWorker.onmessage = ({data}) => {
            switch (data.type) {
                case "training_data":
                    this.generatedImage.draw(data.generatedData, data.width, data.height);
                    this.trainingImage.draw(data.trainingData, data.width, data.height);

                    this.generatorNn.drawSnapshot(data.gSnapshot);
                    this.discriminatorNn.drawSnapshot(data.dSnapshot);
                    break;
            }
        }
    }

    refresh() {

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

            const files = Object.values(loaded.files)
                .map(f => ({match: f.name.match(/^(\d+)-?(\d+)?\.png$/), f}))
                .filter(p => p.match && p.match.length > 1)
                .map(p => {
                    return {
                        // @ts-ignore: Already filtered!
                        index: p.match[1],
                        // @ts-ignore
                        input: p.match[2] || null,
                        file: p.f
                    }
                });

            const result: TrainingData[] = new Array(files.length);
            for (let i = 0; i < files.length; i++) {
                const f = files[i];
                const trainingData = await image.getTrainingDataFromImage(await f.file.async("arraybuffer"))


                const input = matrix.zero(10);
                input[f.input ? Number.parseInt(f.input) : 0] = 1.0;

                //TODO: get proper metadata
                result[i] = {
                    inputSize: 10,
                    input: input,
                    data: trainingData
                };
            }

            this.nnWorker.postMessage({type: "set_data", data: result});
        } finally {
            this.fileLoading = false;
        }
    }
}