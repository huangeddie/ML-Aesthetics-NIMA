from models.densenet import DenseNet
from models.vector_quantization import VQModel
import argparse

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--epochs', '-e', type=int, help='Specify number of epochs', default=1)
parser.add_argument('--version_name', '-v', type=str, default='default', help='Specify name of version')

available_models = list(map(lambda model: model.__name__, [DenseNet, VQModel]))
parser.add_argument('--model', '-m', type=str, help='Specify name of model: {}'.format(available_models))

args = parser.parse_args()

epochs = args.epochs
vers_name = args.version_name
model_name = args.model

Model = eval(model_name)

model = Model()
model.load()
model.train(epochs)
model.checkpoint(vers_name)
