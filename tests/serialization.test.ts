import * as ModelsMock from "./mock/models";
import * as ArrayUtils from "./utils/array";

import {
    Activations,
    ChainModel,
    ChainSerialization,
    Dense,
    GanSerialization,
    Layers,
    Matrix,
    ModelSerialization,
    SequentialModel,
    UniversalModelSerializer
} from "../src/app/neural-network/neural-network";

import {
    BinarySerializer,
    SerializationMetaHeader,
    TensorConfigHeader,
    TensorType
} from "../src/app/neural-network/serialization/binary";
import {SerializationUtils} from "../src/app/neural-network/serialization/utils";
import {ChunkedArrayBuffer} from "../src/app/neural-network/utils/array-buffer";

describe("Should correctly serialize model", () => {
    test("Sequential", () => {
        const model = ModelsMock.sequential();
        model.compile();

        const sModel = ModelSerialization.load(ModelSerialization.save(model));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(model);
    });

    test("Chain", () => {
        const chainModel = ModelsMock.chain();
        chainModel.compile();

        const sModel = ChainSerialization.load(ChainSerialization.save(chainModel));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.models[0].layers) (layer as Dense).skipWeightsInitialization = false;
        for (const layer of sModel.models[1].layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(chainModel);
    });

    test("GAN", () => {
        const ganModel = ModelsMock.gan();

        const sModel = GanSerialization.load(GanSerialization.save(ganModel));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.generator.layers) (layer as Dense).skipWeightsInitialization = false;
        for (const layer of sModel.discriminator.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(ganModel);
    });
});

describe("Should correctly deserialize through UniversalSerializer", () => {
    test("Sequential", () => {
        const model = ModelsMock.sequential();
        model.compile();

        const sModel = UniversalModelSerializer.load(UniversalModelSerializer.save(model));

        // skipWeightsInitialization always has different value, so reset it
        for (const layer of sModel.layers) (layer as Dense).skipWeightsInitialization = false;

        expect(sModel).toEqual(model);
    });

    test("Chain", () => {
        const chainModel = ModelsMock.chain();
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
        const chainModel = ModelsMock.chain();
        chainModel.compile();

        ModelSerialization.load(ChainSerialization.save(chainModel) as any);
    });

    test.failing("Wrong model load #2", () => {
        ChainSerialization.load(GanSerialization.save(ModelsMock.gan()) as any);
    })

    test.failing("Wrong model load #3", () => {
        const model = ModelsMock.sequential();
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
        const chainModel = ModelsMock.chain();
        chainModel.compile();

        ModelSerialization.save(chainModel as any);
    });

    test.failing("Wrong model save #2", () => {
        const chainModel = ModelsMock.chain();
        chainModel.compile();

        ModelSerialization.save(ModelsMock.gan() as any);
    });

    test.failing("Wrong model save #3", () => {
        ModelSerialization.save(ModelsMock.gan() as any);
    });

    test.failing("Wrong model save #4", () => {
        const chainModel = ModelsMock.chain();
        chainModel.compile();

        GanSerialization.save(chainModel as any);
    });

    test.failing("Wrong model save #3", () => {
        const model = ModelsMock.sequential()

        ChainSerialization.save(model as any);
    });

    test.failing.each([ModelsMock.sequential, ModelsMock.chain])
    ("Model should be compiled: %p", (modelFn) => {
        const model = modelFn();
        UniversalModelSerializer.save(model);
    })
})


describe("Binary serialization", () => {
    test("Should correctly serialize to binary format", () => {
        const model = new SequentialModel()
            .addLayer(new Dense(2))
            .addLayer(new Dense(3, {activation: "relu"}))
            .addLayer(new Dense(4, {activation: "tanh"}))
            .addLayer(new Dense(5, {activation: "sigmoid"}));

        model.compile();

        const serialized = BinarySerializer.save(model, TensorType.F64);

        const metaSize = Number(new BigInt64Array(serialized, 0, 1)[0]);
        const header = new Uint8Array(serialized, 8, metaSize);

        const headerObj = JSON.parse(new TextDecoder().decode(header));
        const headerMeta = (headerObj as SerializationMetaHeader).__metadata__;
        const headerTensors = (headerObj as TensorConfigHeader);

        const dataOffset = BigInt64Array.BYTES_PER_ELEMENT + metaSize;
        const dataBuffer = new ChunkedArrayBuffer([serialized])
            .createTypedArray(Float64Array, dataOffset);

        expect(headerMeta.version).toBe("1.0.0");
        expect(headerMeta.model).toBe("Sequential");
        expect(headerMeta.layers.length).toBe(model.layers.length);

        for (let i = 0; i < headerMeta.layers.length; i++) {
            const layer = model.layers[i];
            const layerMeta = headerMeta.layers[i];

            expect(layerMeta.size).toBe(model.layers[i].size);
            expect(layerMeta.key).toBe(SerializationUtils.getTypeAlias(Layers, layer).key);
            expect(layerMeta.activation.key).toBe(SerializationUtils.getTypeAlias(Activations, layer.activation).key);

            const biasesMeta = headerTensors[layerMeta.biasesKey];
            expect(biasesMeta.dtype).toBe(TensorType.F64)
            expect(biasesMeta.shape).toStrictEqual([layer.biases.length]);

            const biases = dataBuffer.subarray(
                biasesMeta.offsets[0] / Float64Array.BYTES_PER_ELEMENT,
                biasesMeta.offsets[1] / Float64Array.BYTES_PER_ELEMENT
            );
            ArrayUtils.arrayCloseTo(biases, layer.biases);

            const weightsMeta = headerTensors[layerMeta.weightsKey];
            expect(weightsMeta.dtype).toBe(TensorType.F64)
            expect(weightsMeta.shape).toStrictEqual([layer.weights.length, layer.weights[0]?.length ?? 0]);

            const weightsData = dataBuffer.subarray(
                weightsMeta.offsets[0] / Float64Array.BYTES_PER_ELEMENT,
                weightsMeta.offsets[1] / Float64Array.BYTES_PER_ELEMENT
            );
            const weights = Matrix.fill(
                i => weightsData.subarray(i * layer.weights[0].length, (i + 1) * layer.weights[0].length),
                layer.weights.length
            );

            ArrayUtils.arrayCloseTo_2d(weights, layer.weights);
        }
    });

    describe("should correctly deserialize from binary format", () => {
        test.each(Object.values(TensorType))
        ("%p", (tensorType) => {
            const model = ModelsMock.sequential();
            model.compile();

            const sModel = BinarySerializer.load(BinarySerializer.save(model, TensorType.F64));
            for (let i = 0; i < model.layers.length; i++) {
                expect(sModel.layers[i].activation).toEqual(model.layers[i].activation);
                ArrayUtils.arrayCloseTo(sModel.layers[i].biases, model.layers[i].biases)
                ArrayUtils.arrayCloseTo_2d(sModel.layers[i].weights, model.layers[i].weights)
            }
        })
    })

    test.failing("Should fail if model isn't compiled", () => {
        const model = new SequentialModel()
            .addLayer(new Dense(2))
            .addLayer(new Dense(3));

        BinarySerializer.save(model);
    })
})