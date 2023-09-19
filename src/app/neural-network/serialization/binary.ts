import {ILoss, IModel, IOptimizer} from "../engine/base";
import {OptimizerT} from "../engine/optimizers";
import {LossT} from "../engine/loss";
import {Matrix1D, Matrix2D} from "../engine/matrix";
import {ChunkedArrayBuffer, TypedArray, TypedArrayT} from "../utils/array-buffer";
import {Activations, Layers, Matrix, Models, ModelSerialization} from "../neural-network";

import {SerializationEntry, SerializedParams} from "./base";
import {SerializationUtils} from "./utils";

/*
    TODO: ALWAYS CHANGE VERSION
*/

const Major = 1;
const Minor = 0;
const Patch = 0;

const Version = [Major, Minor, Patch].join(".");

export type LayerMeta = {
    key: keyof typeof Layers,
    size: number,
    activation: SerializationEntry<typeof Activations>,
    params: SerializedParams,
    weightsKey: string,
    biasesKey: string
}

export type SerializationMetadata = {
    version: string,
    model: keyof typeof Models,
    layers: LayerMeta[]
}

export enum TensorType {
    "F64" = "F64",
    "F32" = "F32",
}

export type TensorConfig = {
    dtype: TensorType,
    shape: number[],
    offsets: [number, number]
}

export type SerializationMetaHeader = {
    __metadata__: SerializationMetadata
}

export type TensorConfigHeader = {
    [name: string]: TensorConfig
}

export class BinarySerializer {
    static save(model: IModel, dataType = TensorType.F32): ArrayBuffer {
        if (!model.isCompiled) throw new Error("Model should be compiled");

        const tensorsHeader: TensorConfigHeader = {};
        const layersMeta: LayerMeta[] = new Array(model.layers.length);
        const dataChunks = [];
        let offset = 0;

        for (let i = 0; i < model.layers.length; i++) {
            const layer = model.layers[i];

            const layerMeta = {
                key: SerializationUtils.getTypeAlias(Layers, layer).key,
                size: layer.size,
                activation: ModelSerialization.saveActivation(layer.activation),
                params: SerializationUtils.getSerializableParams(layer),
                weightsKey: `weights_${i}`,
                biasesKey: `biases_${i}`
            };

            layersMeta[i] = layerMeta;

            const biasesChunk = this._getBinaryRepresentation1d(layer.biases, dataType);
            const biasesChunkLength = biasesChunk.length * biasesChunk.BYTES_PER_ELEMENT;
            tensorsHeader[layerMeta.biasesKey] = {
                dtype: dataType,
                shape: [layer.biases.length],
                offsets: [offset, offset + biasesChunkLength]
            }

            dataChunks.push(biasesChunk);
            offset += biasesChunkLength;

            const weightsChunks = this._getBinaryRepresentation2d(layer.weights, dataType);
            const weightsChunkSize = weightsChunks.length * (weightsChunks[0]?.length ?? 0) * (weightsChunks[0]?.BYTES_PER_ELEMENT ?? 0);
            tensorsHeader[layerMeta.weightsKey] = {
                dtype: dataType,
                shape: [layer.weights.length, layer.weights[0]?.length ?? 0],
                offsets: [offset, offset + weightsChunkSize]
            }

            dataChunks.push(...weightsChunks);
            offset += weightsChunkSize;
        }

        const metadata: SerializationMetadata = {
            version: Version,
            model: "Sequential",
            layers: layersMeta
        }

        const metaHeader: SerializationMetaHeader = {
            __metadata__: metadata,
        }

        const header = {
            ...metaHeader,
            ...tensorsHeader
        }

        const headerBytes = new TextEncoder().encode(JSON.stringify(header));
        const headerSize = new BigInt64Array(1);
        headerSize[0] = BigInt(headerBytes.length);

        const resultChunks = [headerSize, headerBytes].concat(dataChunks);
        const chunkedArray = new ChunkedArrayBuffer(resultChunks.map(c => c.buffer));
        return chunkedArray.toTypedArray(Uint8Array).buffer;
    }

    static load(data: ArrayBuffer, optimizer?: OptimizerT | IOptimizer, loss?: LossT | ILoss): IModel {
        const metaSize = Number(new BigInt64Array(data, 0, 1)[0]);
        const header = new Uint8Array(data, BigInt64Array.BYTES_PER_ELEMENT, metaSize);

        const headerObj = JSON.parse(new TextDecoder().decode(header));
        const headerMeta = (headerObj as SerializationMetaHeader).__metadata__;
        const headerTensors = (headerObj as TensorConfigHeader);

        const [major, minor, patch] = headerMeta.version.split(".").map(s => Number.parseInt(s));
        if (major !== Major || minor > Minor)
            throw new Error(`Unsupported version ${major}.${minor}.${patch}. Supported versions: ${Major}.0.* – ${Major}.${Minor}.*`)

        const modelT = Models[headerMeta.model];
        if (!modelT) throw new Error(`Invalid model: ${headerMeta.model}`);

        const model = new modelT(optimizer, loss);

        const dataOffset = BigInt64Array.BYTES_PER_ELEMENT + metaSize;
        const tensorsDataArray = new ChunkedArrayBuffer([data]).createTypedArray(Float64Array, dataOffset);

        let layerIndex = 0;
        for (const layerConf of headerMeta.layers) {
            const biasesMeta = headerTensors[layerConf.biasesKey];
            const biases = tensorsDataArray.subarray(
                biasesMeta.offsets[0] / Float64Array.BYTES_PER_ELEMENT,
                biasesMeta.offsets[1] / Float64Array.BYTES_PER_ELEMENT
            );

            const weightsMeta = headerTensors[layerConf.weightsKey];
            const weightsData = tensorsDataArray.subarray(
                weightsMeta.offsets[0] / Float64Array.BYTES_PER_ELEMENT,
                weightsMeta.offsets[1] / Float64Array.BYTES_PER_ELEMENT
            );

            const prevSize = layerIndex > 0 ? headerMeta.layers[layerIndex - 1].size : 0
            const weights = Matrix.fill(
                i => weightsData.subarray(i * prevSize, (i + 1) * prevSize),
                layerConf.size
            );

            const layer = ModelSerialization.loadLayer({
                ...layerConf,
                biases: biases,
                weights: weights,
                biasInitializer: "zero",
                weightInitializer: "zero"
            }, layerIndex++)

            model.addLayer(layer);
        }

        model.compile();
        return model;
    }

    private static _getBinaryRepresentation1d(data: Matrix1D, dataType: TensorType) {
        const result = this._getArray(dataType, data.length);
        result.set(data);

        return result;
    }

    private static _getBinaryRepresentation2d(data: Matrix2D, dataType: TensorType) {
        const chunks = new Array(data.length);

        for (let i = 0; i < data.length; i++) {
            const chunk = this._getArray(dataType, data[i].length);
            chunk.set(data[i]);
            chunks[i] = chunk;
        }

        return chunks;
    }

    private static _getArrayT(dataType: TensorType): TypedArrayT<TypedArray> {
        switch (dataType) {
            case TensorType.F32:
                return Float32Array;

            case TensorType.F64:
                return Float64Array;

            default:
                throw new Error(`Unsupported data type: ${dataType}`)
        }
    }

    private static _getArray(dataType: TensorType, size: number) {
        const arrayT = this._getArrayT(dataType)
        return new arrayT(size);
    }
}