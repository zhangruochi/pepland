import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from finetune_example.models.pepland.inference import FeatureExtractor
from finetune_example.models.core import PropertyPredictor


def test_feature_extractor():
    """ Test the feature extractor.
        This model use pepland model to extract features.
    """
    feature_extractor = FeatureExtractor()
    input_smiles = [
        'C/C=C/C[C@@H](C)[C@@H](O)[C@H]1C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C',
        'CC(C)C[C@H]1NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC1=O'
    ]
    pep_embeds = feature_extractor(input_smiles)
    print(pep_embeds.shape)

def test_model_predictor():
    """ Test the model predictor.
        This model use pepland model to extract features and then use a MLP to predict the property.
    """

    model = PropertyPredictor(
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "./models/pepland/cpkt/model"),
        pool_type="avg",
        hidden_dims=[256,128],
        mlp_dropout=0.1
    )
    input_smiles = [
        'C/C=C/C[C@@H](C)[C@@H](O)[C@H]1C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C',
        'CC(C)C[C@H]1NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC1=O'
    ]

    pred = model(input_smiles)
    print(pred)


    ## TODO, use a trainer and a dataset to train the model.

    

if __name__ == "__main__":
    test_feature_extractor()
    test_model_predictor()