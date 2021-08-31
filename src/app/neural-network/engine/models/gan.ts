import * as matrix from "../matrix"

import {SequentialModel} from "./sequential";
import {ChainModel} from "./chain";
import {OptimizerT} from "../optimizers";
import {IOptimizer} from "../base";


export class GenerativeAdversarialModel {
    ganChain: ChainModel;

    constructor(public generator: SequentialModel, public discriminator: SequentialModel, optimizer: OptimizerT | IOptimizer = 'sgd') {
        this.ganChain = new ChainModel(optimizer);
        this.ganChain
            .addModel(generator)
            .addModel(discriminator, false)
            .compile();
    }

    compute(input: matrix.Matrix1D): matrix.Matrix1D {
        return this.generator.compute(input);
    }

    train(real: matrix.Matrix1D, realExpected: matrix.Matrix1D, input: matrix.Matrix1D, inputExpected: matrix.Matrix1D) {
        this.discriminator.train(real, realExpected);

        const fake = this.generator.compute(input);
        this.discriminator.train(fake, inputExpected);

        this.ganChain.train(input, realExpected);
    }
}