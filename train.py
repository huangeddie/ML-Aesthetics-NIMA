from models.densenet import DenseNet
from models.vector_quantization import VQModel
import argparse

available_models = list(map(lambda model: model.__name__, [DenseNet, VQModel]))

# Parse arguments
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--epochs', '-e', type=int, help='Number of epochs', default=1)
parser.add_argument('--version_name', '-v', type=str, default='default', help='Name of version')
parser.add_argument('--model', '-m', type=str, help='Choose model: {}'.format(available_models))
args = parser.parse_args()

epochs = args.epochs
vers_name = args.version_name
model_name = args.model

# Do stuff
Model = eval(model_name)

model = Model()
model.load()
model.train(epochs)
model.checkpoint(vers_name)
