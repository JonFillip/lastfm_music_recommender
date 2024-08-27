from kfp.v2.dsl import component
from kfp.v2.dsl import Input, Output, Model

@component(
    packages_to_install=['pandas', 'scikit-learn', 'tensorflow'],
    base_image='python:3.9'
)
def train_model(
    algorithm: str,
    input_data: Input["Dataset"],
    output_model: Output[Model]
):
    import pandas as pd
    import pickle
    
    if algorithm == 'content_based':
        from src.algorithms.content_based import ContentBasedRecommender
        model = ContentBasedRecommender()
    elif algorithm == 'collaborative':
        from src.algorithms.collaborative_filtering import CollaborativeFilter
        model = CollaborativeFilter()
    elif algorithm == 'hybrid':
        from src.algorithms.hybrid_model import HybridRecommender
        model = HybridRecommender()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    data = pd.read_csv(input_data.path)
    model.fit(data)
    
    with open(output_model.path, 'wb') as f:
        pickle.dump(model, f)