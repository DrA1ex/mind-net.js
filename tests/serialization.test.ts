import {
    ChainModel, Dense, GanSerialization, ModelSerialization, ChainSerialization, UniversalModelSerializer
} from "../src/app/neural-network/neural-network";

import * as Models from "./mock/models";

describe("Should correctly serialize model", () => {
    test("Sequential", () => {
        const model = Models.sequential();
        model.compile();

        const sModel = ModelSerialization.load(ModelSerialization.save(model));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(model);
    });

    test("Chain", () => {
        const chainModel = Models.chain();
        chainModel.compile();

        const sModel = ChainSerialization.load(ChainSerialization.save(chainModel));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.models[0].layers) (layer as Dense).skipWeightsInitialization = false;
        for (const layer of sModel.models[1].layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(chainModel);
    });

    test("GAN", () => {
        const ganModel = Models.gan();

        const sModel = GanSerialization.load(GanSerialization.save(ganModel));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.generator.layers) (layer as Dense).skipWeightsInitialization = false;
        for (const layer of sModel.discriminator.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(ganModel);
    });
});

describe("Should correctly deserialize through UniversalSerializer", () => {
    test("Sequential", () => {
        const model = Models.sequential();
        model.compile();

        const sModel = UniversalModelSerializer.load(UniversalModelSerializer.save(model));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(model);
    });

    test("Chain", () => {
        const chainModel = Models.chain();
        chainModel.compile();

        const sModel = UniversalModelSerializer.load(UniversalModelSerializer.save(chainModel)) as ChainModel;

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.models[0].layers) (layer as Dense).skipWeightsInitialization = false;
        for (const layer of sModel.models[1].layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(chainModel);
    });
});

describe("Should fail when invalid data passed", () => {
    test.failing("Wrong model load #1", () => {
        const chainModel = Models.chain();
        chainModel.compile();

        ModelSerialization.load(ChainSerialization.save(chainModel) as any);
    });

    test.failing("Wrong model load #2", () => {
        ChainSerialization.load(GanSerialization.save(Models.gan()) as any);
    })

    test.failing("Wrong model load #3", () => {
        const model = Models.sequential();
        model.compile();

        GanSerialization.load(ModelSerialization.save(model) as any);
    })

    test.failing("Wrong model load #4", () => {
        ChainSerialization.load({} as any);
    })

    test.failing("Wrong model load #5", () => {
        ChainSerialization.load(undefined as any);
    })

    test.failing("Wrong model save #1", () => {
        const chainModel = Models.chain();
        chainModel.compile();

        ModelSerialization.save(chainModel as any);
    });

    test.failing("Wrong model save #2", () => {
        const chainModel = Models.chain();
        chainModel.compile();

        ModelSerialization.save(Models.gan() as any);
    });

    test.failing("Wrong model save #3", () => {
        ModelSerialization.save(Models.gan() as any);
    });

    test.failing("Wrong model save #4", () => {
        const chainModel = Models.chain();
        chainModel.compile();

        GanSerialization.save(chainModel as any);
    });

    test.failing("Wrong model save #3", () => {
        const model = Models.sequential()

        ChainSerialization.save(model as any);
    });

    test.failing.each([Models.sequential, Models.chain])
    ("Model should be compiled: %p", (modelFn) => {
        const model = modelFn();
        UniversalModelSerializer.save(model);
    })
})
