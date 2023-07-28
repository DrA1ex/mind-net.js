import {SequentialModel} from "./sequential";
import {ChainModel} from "./chain";
import {GenerativeAdversarialModel} from "./gan";

export const Models = {
    Sequential: SequentialModel,
    Chain: ChainModel,
    GAN: GenerativeAdversarialModel
}