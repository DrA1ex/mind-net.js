import {
    AdamOptimizer, ChainModel,
    Dense, GanSerialization, GenerativeAdversarialModel,
    LeakyReluActivation, MeanAbsoluteErrorLoss,
    ModelSerialization, RMSPropOptimizer,
    SequentialModel, SgdOptimizer,
    Initializers, ChainSerialization, UniversalModelSerializer
} from "../src/app/neural-network/neural-network";


function _sequential() {
    const model = new SequentialModel(
        new AdamOptimizer({lr: 0.1, decay: 0.2, beta1: 0.3, beta2: 0.4, eps: 0.5}),
        new MeanAbsoluteErrorLoss({k: 123})
    );
    model.addLayer(new Dense(4));
    model.addLayer(new Dense(5, {activation: "leakyRelu"}));
    model.addLayer(new Dense(6, {activation: "sigmoid", options: {dropout: 0.1, l2WeightRegularization: 0.2}}));
    model.addLayer(new Dense(7, {activation: "tanh", options: {l1WeightRegularization: 0.1}}));
    model.addLayer(new Dense(14, {
        weightInitializer: "normal", biasInitializer: "uniform",
        options: {l2BiasRegularization: 0.3, l1BiasRegularization: 0.4}
    }));
    model.addLayer(new Dense(8, {activation: new LeakyReluActivation({alpha: 0.15})}));
    model.addLayer(new Dense(9, {
        weightInitializer: Initializers.xavier, biasInitializer: Initializers.xavier_normal
    }));

    return model;
}

function _chain() {
    const model1 = new SequentialModel(
        new AdamOptimizer({lr: 0.1, decay: 0.2, beta1: 0.3, beta2: 0.4, eps: 0.5}),
        new MeanAbsoluteErrorLoss({k: 123})
    );
    model1.addLayer(new Dense(4));
    model1.addLayer(new Dense(5, {activation: "leakyRelu"}));
    model1.addLayer(new Dense(6, {activation: "sigmoid", options: {dropout: 0.1, l2WeightRegularization: 0.2}}));

    const model2 = new SequentialModel(
        new RMSPropOptimizer({lr: 0.4, decay: 0.3, beta: 0.2, eps: 0.1}),
        new MeanAbsoluteErrorLoss({k: 321})
    )
    model2.addLayer(new Dense(6, {activation: "tanh", options: {l1WeightRegularization: 0.1}}));
    model2.addLayer(new Dense(7, {
        weightInitializer: "normal", biasInitializer: "uniform",
        options: {l2BiasRegularization: 0.3, l1BiasRegularization: 0.4}
    }));
    model2.addLayer(new Dense(8, {activation: new LeakyReluActivation({alpha: 0.15})}));
    model2.addLayer(new Dense(9, {
        weightInitializer: Initializers.xavier, biasInitializer: Initializers.xavier_normal
    }));

    const chainModel = new ChainModel(new SgdOptimizer({lr: 1}), "binaryCrossEntropy");
    chainModel.addModel(model1);
    chainModel.addModel(model2, false);

    return chainModel;
}

function _gan() {
    const model1 = new SequentialModel(
        new AdamOptimizer({lr: 0.1, decay: 0.2, beta1: 0.3, beta2: 0.4, eps: 0.5}),
        new MeanAbsoluteErrorLoss({k: 123})
    );
    model1.addLayer(new Dense(4));
    model1.addLayer(new Dense(5, {activation: "leakyRelu"}));
    model1.addLayer(new Dense(6, {activation: "sigmoid", options: {dropout: 0.1, l2WeightRegularization: 0.2}}));
    model1.compile();

    const model2 = new SequentialModel(
        new RMSPropOptimizer({lr: 0.4, decay: 0.3, beta: 0.2, eps: 0.1}),
        new MeanAbsoluteErrorLoss({k: 321})
    )
    model2.addLayer(new Dense(6, {activation: "tanh", options: {l1WeightRegularization: 0.1}}));
    model2.addLayer(new Dense(7, {
        weightInitializer: "normal", biasInitializer: "uniform",
        options: {l2BiasRegularization: 0.3, l1BiasRegularization: 0.4}
    }));
    model2.addLayer(new Dense(8, {activation: new LeakyReluActivation({alpha: 0.15})}));
    model2.addLayer(new Dense(9, {
        weightInitializer: Initializers.xavier, biasInitializer: Initializers.xavier_normal
    }));
    model2.addLayer(new Dense(1));
    model2.compile();

    return new GenerativeAdversarialModel(model1, model2);
}

describe("Should correctly serialize model", () => {
    test("Sequential", () => {
        const model = _sequential();
        model.compile();

        const sModel = ModelSerialization.load(ModelSerialization.save(model));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(model);
    });

    test("Chain", () => {
        const chainModel = _chain();
        chainModel.compile();

        const sModel = ChainSerialization.load(ChainSerialization.save(chainModel));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.models[0].layers) (layer as Dense).skipWeightsInitialization = false;
        for (const layer of sModel.models[1].layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(chainModel);
    });

    test("GAN", () => {
        const ganModel = _gan();

        const sModel = GanSerialization.load(GanSerialization.save(ganModel));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.generator.layers) (layer as Dense).skipWeightsInitialization = false;
        for (const layer of sModel.discriminator.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(ganModel);
    });
});

describe("Should correctly deserialize through UniversalSerializer", () => {
    test("Sequential", () => {
        const model = _sequential();
        model.compile();

        const sModel = UniversalModelSerializer.load(UniversalModelSerializer.save(model));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(model);
    });

    test("Chain", () => {
        const chainModel = _chain();
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
        const chainModel = _chain();
        chainModel.compile();

        ModelSerialization.load(ChainSerialization.save(chainModel) as any);
    });

    test.failing("Wrong model load #2", () => {
        ChainSerialization.load(GanSerialization.save(_gan()) as any);
    })

    test.failing("Wrong model load #3", () => {
        const model = _sequential();
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
        const chainModel = _chain();
        chainModel.compile();

        ModelSerialization.save(chainModel as any);
    });

    test.failing("Wrong model save #2", () => {
        const chainModel = _chain();
        chainModel.compile();

        ModelSerialization.save(_gan() as any);
    });

    test.failing("Wrong model save #3", () => {
        ModelSerialization.save(_gan() as any);
    });

    test.failing("Wrong model save #4", () => {
        const chainModel = _chain();
        chainModel.compile();

        GanSerialization.save(chainModel as any);
    });

    test.failing("Wrong model save #3", () => {
        const model = _sequential()

        ChainSerialization.save(model as any);
    });
})
