import torch
import torchvision.models as models 

#================================================================#

def get_efficientnet_model(model_name: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    """
    Returns the specified EfficientNet model.

    Args:
    - model_name (str): name of the model (e.g., 'efficientnet_b0' to 'efficientnet_b7').
    - num_classes (int): number of classes for the final layer.
    - pretrained (bool): if True, returns a model pre-trained on ImageNet (1000 classes).

    Returns:
    - model: A PyTorch model.
    """

    assert model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
                          'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'], ValueError("Unsupported EfficientNet version. Choose from 'efficientnet_b0' to 'efficientnet_b7'.")
    
    if pretrained and num_classes != 1000:
        model = models.efficientnet.__dict__[model_name](weights=models.EfficientNet_B3_Weights.DEFAULT)  
    else:
        model = models.efficientnet.__dict__[model_name]()

    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    return model