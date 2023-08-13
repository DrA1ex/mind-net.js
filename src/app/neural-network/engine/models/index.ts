import {SequentialModel} from "./sequential";
import {ChainModel} from "./chain";
import {GenerativeAdversarialModel} from "./gan";

export const Models = {
    Sequential: SequentialModel,
}

export const ComplexModels = {
    Chain: ChainModel,
    GenerativeAdversarial: GenerativeAdversarialModel
}