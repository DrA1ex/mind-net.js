import {SetupMockRandom} from "./mock/common";
import * as WorkerMock from "./mock/worker";
import * as Models from "./mock/models";

import {
    GanSerialization,
    ModelSerialization,
    ParallelGanWrapper,
    ParallelModelWrapper
} from "../src/app/neural-network/neural-network";
import {ParallelWorkerImpl} from "../src/app/neural-network/engine/wrapper/parallel.worker.impl";

import {RandomMockData, TrainData, TrainExpected, TestInput} from "./fixture/parallel"

const randomMock = SetupMockRandom(RandomMockData, true);
const BatchSize = 32;

Models.disableProgressLogs();

describe("Should correct train network in parallel", () => {
    beforeEach(() => randomMock.reset());
    WorkerMock.mockWorker(ParallelWorkerImpl);

    describe("Sequential", () => {
        test.each([1, 2, 4, 16, 32])
        ("parallelism: %d", async (parallelism) => {
            const model = Models.sequential();
            model.compile();

            const copy = ModelSerialization.load(ModelSerialization.save(model));
            const pModel = new ParallelModelWrapper(copy, parallelism);
            await pModel.init();

            randomMock.reset();
            model.train(TrainData, TrainExpected, {batchSize: BatchSize});

            randomMock.reset();
            await pModel.train(TrainData, TrainExpected, {batchSize: BatchSize / parallelism});

            const modelOut = model.compute(TestInput);
            const pModelOut = await pModel.compute([TestInput]);

            expect(pModelOut[0]).toStrictEqual(modelOut);
        });
    })

    describe("GAN", () => {
        test.each([1, 2, 4, 16, 32])
        ("parallelism: %d", async (parallelism) => {
            const model = Models.gan(4, 4);

            const copy = GanSerialization.load(GanSerialization.save(model));
            const pModel = new ParallelGanWrapper(copy, parallelism);
            await pModel.init();

            randomMock.reset();
            model.train(TrainData, {batchSize: BatchSize});

            randomMock.reset();
            await pModel.train(TrainData, {batchSize: BatchSize});

            const modelOut = model.generator.compute(TestInput);
            const pModelOut = await pModel.generatorWrapper.compute([TestInput]);

            expect(pModelOut[0]).toStrictEqual(modelOut);
        });
    });
});