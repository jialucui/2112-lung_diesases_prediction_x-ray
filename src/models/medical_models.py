class MultiTaskPneumoniaModel:
    def __init__(self, num_classes, model_type='resnet50'): 
        self.model_type = model_type
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        if self.model_type == 'resnet50':
            return self.create_resnet50()
        elif self.model_type == 'densenet121':
            return self.create_densenet121()
        elif self.model_type == 'efficientnet-b0':
            return self.create_efficientnet_b0()
        else:
            raise ValueError('Invalid model type')

    def create_resnet50(self):
        # Code to create ResNet50 model
        pass

    def create_densenet121(self):
        # Code to create DenseNet121 model
        pass

    def create_efficientnet_b0(self):
        # Code to create EfficientNet-B0 model
        pass

class SingleTaskPneumoniaModel(MultiTaskPneumoniaModel):
    def __init__(self, model_type='resnet50'):
        super().__init__(num_classes=1, model_type=model_type)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

