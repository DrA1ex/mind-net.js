import * as matrix from "../matrix"
import * as iter from "../iter"

import {SequentialModel} from "./sequential";
import {ChainModel} from "./chain";
import {OptimizerT} from "../optimizers";
import {ILoss, IOptimizer} from "../base";
import {LossT} from "../loss";


export class GenerativeAdversarialModel {
    ganChain: ChainModel;

    get optimizer() {return this.ganChain.models[0].optimizer;};

    constructor(public generator: SequentialModel,
                public discriminator: SequentialModel,
                optimizer: OptimizerT | IOptimizer = 'sgd',
                loss: LossT | ILoss = "mse") {
        if (discriminator.layers[discriminator.layers.length - 1].size !== 1) {
            throw new Error("Size of discriminator's output should be 1");
        }

        this.ganChain = new ChainModel(optimizer, loss);
        this.ganChain
            .addModel(generator)
            .addModel(discriminator, false)
            .compile();
    }

    public compute(input: matrix.Matrix1D): matrix.Matrix1D {
        return this.generator.compute(input);
    }

    public train(real: matrix.Matrix1D[], batchSize: number = 32) {
        const shuffledTrainSet = iter.shuffled(real);
        for (const batch of iter.partition(shuffledTrainSet, batchSize)) {
            this.trainBatch(batch);
        }
    }

    public trainBatch(batch: matrix.Matrix1D[]) {
        const ones = matrix.fill_value([1], batch.length);
        const almostOnes = matrix.fill_value([0.9], batch.length);
        const zeros = matrix.fill_value([0], batch.length);
        const noise = matrix.random_2d(batch.length, this.generator.layers[0].size, -1, 1);
        const fake = noise.map(input => this.generator.compute(input));

        this.discriminator.trainBatch(iter.zip([...batch, ...fake], [...almostOnes, ...zeros]));
        this.ganChain.trainBatch(iter.zip(noise, ones));
    }
}