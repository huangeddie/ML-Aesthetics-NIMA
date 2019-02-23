import models.vector_quantization

models_to_train = [
    models.vector_quantization.VQModel,
]


for Model in models_to_train:
    model = Model()
    model.load()
    model.train()
    version = model.checkpoint()
    print('Saved {} to version {}'.format(Model.__name__, version))