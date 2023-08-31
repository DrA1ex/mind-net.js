import {
    AdamOptimizer, ChainModel, Dense, GenerativeAdversarialModel, LeakyReluActivation, MeanAbsoluteErrorLoss,
    RMSPropOptimizer, SequentialModel, SgdOptimizer, Initializers
} from "../../src/app/neural-network/neural-network";

export function sequential(inSize = 4, outSize = 10) {
    const model = new SequentialModel(
        new AdamOptimizer({lr: 0.1, decay: 0.2, beta1: 0.3, beta2: 0.4, eps: 0.5}),
        new MeanAbsoluteErrorLoss({k: 123})
    );
    model.addLayer(new Dense(inSize));
    model.addLayer(new Dense(5, {activation: "leakyRelu"}));
    model.addLayer(new Dense(6, {activation: "sigmoid", options: {dropout: 0.1, l2WeightRegularization: 0.2}}));
    model.addLayer(new Dense(7, {activation: "tanh", options: {l1WeightRegularization: 0.1}}));
    model.addLayer(new Dense(14, {
        weightInitializer: "normal", biasInitializer: "uniform",
        options: {l2BiasRegularization: 0.3, l1BiasRegularization: 0.4}
    }));
    model.addLayer(new Dense(8, {activation: new LeakyReluActivation({alpha: 0.15})}));
    model.addLayer(new Dense(outSize, {
        weightInitializer: Initializers.xavier, biasInitializer: Initializers.xavier_normal
    }));

    return model;
}

export function chain(inSize = 4, outSize = 10) {
    const model1 = new SequentialModel(
        new AdamOptimizer({lr: 0.1, decay: 0.2, beta1: 0.3, beta2: 0.4, eps: 0.5}),
        new MeanAbsoluteErrorLoss({k: 123})
    );
    model1.addLayer(new Dense(inSize));
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
    model2.addLayer(new Dense(outSize, {
        weightInitializer: Initializers.xavier, biasInitializer: Initializers.xavier_normal
    }));

    const chainModel = new ChainModel(new SgdOptimizer({lr: 1}), "binaryCrossEntropy");
    chainModel.addModel(model1);
    chainModel.addModel(model2, false);

    return chainModel;
}

export function gan(inSize = 10, outSize = 4) {
    const model1 = new SequentialModel(
        new AdamOptimizer({lr: 0.1, decay: 0.2, beta1: 0.3, beta2: 0.4, eps: 0.5}),
        new MeanAbsoluteErrorLoss({k: 123})
    );

    model1.addLayer(new Dense(inSize));
    model1.addLayer(new Dense(5, {activation: "leakyRelu"}));
    model1.addLayer(new Dense(outSize, {activation: "sigmoid", options: {dropout: 0.1, l2WeightRegularization: 0.2}}));
    model1.compile();

    const model2 = new SequentialModel(
        new RMSPropOptimizer({lr: 0.4, decay: 0.3, beta: 0.2, eps: 0.1}),
        new MeanAbsoluteErrorLoss({k: 321})
    );

    model2.addLayer(new Dense(outSize, {activation: "tanh", options: {l1WeightRegularization: 0.1}}));
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