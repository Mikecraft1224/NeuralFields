from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
import sys, os
import time

from toGif import toGif

# FLAGS
NOGRAPH = False
HIDDEN = 256
LAYERS = 8

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
class NeuralField(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int, encdim: int):
        super(NeuralField, self).__init__()

        encInput = input_size * 2 * encdim if encdim > 0 else input_size
        self.input = nn.Linear(encInput, hidden_size) # * 2 * encdim
        self.layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.input(x))
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.output(x))
        return x

    def createData(img: Image, name: int, res: int, encdim: int) -> tuple[Tensor, Tensor]:
        img = img.resize((res, res))
        img.save(f"{name}/base_{name}_{res}.png")
        img = np.array(img) / 255.0
        
        xs, ys = np.linspace(0, 1, res), np.linspace(0, 1, res)
        xs, ys = np.meshgrid(xs, ys)
        coords = np.vstack([xs.ravel(), ys.ravel()]).T

        # Positional encoding
        if encdim > 0:
            encoded = []
            for i in range(encdim):
                freq = 2 ** i
                encoded.append(np.sin(freq * np.pi * coords))
                encoded.append(np.cos(freq * np.pi * coords))
            coords = np.concatenate(encoded, axis=-1)

            # print(f"Shape: {coords.shape}")

        colors = img.reshape(-1, 3)
        return (
            torch.tensor(coords, dtype=torch.float32).to(device), # Move to GPU
            torch.tensor(colors, dtype=torch.float32).to(device)  # Move to GPU
        )

    def trainData(img: str, name: str, gens: int, steps: int, res: int = None, input_size=2, hidden_size=HIDDEN, output_size=3, hidden_layers=LAYERS, encdim=2):
        if os.path.exists(name):
            print("Model already exists")
            sys.exit(1)
        os.mkdir(name)

        img = Image.open(img).convert("RGB")
        if res == None:
            res = img.size[0]
            print(f"Resolution not specified, using {res}x{res}")

        model = NeuralField(input_size=input_size, hidden_size=hidden_size, output_size=output_size, hidden_layers=hidden_layers, encdim=encdim).to(device) # Move to GPU
        crit = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        coords, colors = NeuralField.createData(img, name, res, encdim)

        errors = []

        def process_function(res: int, model: NeuralField, epoch: int):
            model.eval()

            xs, ys = np.linspace(0, 1, res), np.linspace(0, 1, res)
            xs, ys = np.meshgrid(xs, ys)
            coords = np.vstack([xs.ravel(), ys.ravel()]).T

            if encdim > 0:
                encoded = []
                for i in range(encdim):
                    freq = 2 ** i
                    encoded.append(np.sin(freq * np.pi * coords))
                    encoded.append(np.cos(freq * np.pi * coords))
                coords = np.concatenate(encoded, axis=-1)
            coords = torch.tensor(coords, dtype=torch.float32).to(device)

            with torch.no_grad():
                out = model(coords).cpu().numpy().reshape(res, res, 3)
            img = Image.fromarray((out * 255).astype(np.uint8))
            img.save(f"{name}/gen_{name}_{epoch}.png")

        for e in range(1, gens):
            model.train()
            opt.zero_grad()
            out = model(coords)
            loss = crit(out, colors)
            loss.backward()
            opt.step()

            errors.append(loss.item())

            if e % 100 == 0 and not NOGRAPH:
                print(f"Epoch {e}: {loss.item()}")
            if e % steps == 0 and steps != 0:
                process_function(res, model, e)

    
        plt.plot(errors, label=f"{name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Over Time")

        if not NOGRAPH:
            plt.show()
            plt.figure()

        torch.save(model.state_dict(), f"{name}/{name}.pt")

        return model, res, loss.item()

    def generateImage(res: int, model=None, modelName=None, show=True, save=True, name="output", encdim=2):
        if model is None:
            model = NeuralField().to(device)
            model.load_state_dict(torch.load(f"{modelName}.pt", weights_only=True))
        model.eval()

        xs, ys = np.linspace(0, 1, res), np.linspace(0, 1, res)
        xs, ys = np.meshgrid(xs, ys)
        coords = np.vstack([xs.ravel(), ys.ravel()]).T

        if encdim > 0:
            encoded = []
            for i in range(encdim):
                freq = 2 ** i
                encoded.append(np.sin(freq * np.pi * coords))
                encoded.append(np.cos(freq * np.pi * coords))
            coords = np.concatenate(encoded, axis=-1)
        coords = torch.tensor(coords, dtype=torch.float32).to(device)

        with torch.no_grad():
            out = model(coords).cpu().numpy().reshape(res, res, 3)

        img = Image.fromarray((out * 255).astype(np.uint8))
        if show:
            img.show()
        if save:
            img.save(f"{name}.png")
        return

# Main
def error():
    print("Usage: python main.py [-t/-g/-q] [-n name] [-i img] [-e epochs] [-s steps] [-r resolution] [-d dimension] [-c config]")
    print("  -t: Train a new model")
    print("     -n: Name of the model")
    print("     -i: Image to train on")
    print("     -e: Number of epochs to train")
    print("     -s: Number of steps to generate images")
    print("     -r: Resolution of the image")
    print("     -d: Dimension of the positional encoding (0 for none)")
    print("  -g: Generate an image from a model")
    print("     -n: Name of the model")
    print("     -r: Resolution of the image")
    print("     -d: Dimension of the positional encoding")
    print("  -q: Generate multiple networks based on config file")
    print("     -c: Config file")
    print()
    print("Example config line:")
    print("name:test img:test.jpg epochs:1000 steps:100 res:256 input:2 hidden:1024 output:3 layers:10")
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        error()

    name: str = None
    img: str = None
    epochs: int = None
    steps: int = None
    res: int = None
    encdim: int = None
    config: str = None

    for i in range(2, len(sys.argv), 2):
        if sys.argv[i] == "-n":
            name = sys.argv[i + 1]
        elif sys.argv[i] == "-i":
            img = sys.argv[i + 1]
        elif sys.argv[i] == "-e":
            epochs = int(sys.argv[i + 1])
        elif sys.argv[i] == "-s":
            steps = int(sys.argv[i + 1])
        elif sys.argv[i] == "-r":
            res = int(sys.argv[i + 1])
        elif sys.argv[i] == "-d":
            encdim = int(sys.argv[i + 1])
        elif sys.argv[i] == "-c":
            config = sys.argv[i + 1]

    if sys.argv[1] == "-t":
        if name is None or epochs is None or steps is None or img is None or encdim is None: error()

        model, res, loss = NeuralField.trainData(img=img, name=name, gens=epochs, steps=steps, res=res, encdim=encdim)
        NeuralField.generateImage(res=res, model=model, name=f"{name}/{name}_{epochs}")
        toGif(name)
    elif sys.argv[1] == "-g":
        if name is None or res is None or encdim is None: error()

        NeuralField.generateImage(res=res, modelName=name, name=f"{name}/{name}", encdim=encdim)
    elif sys.argv[1] == "-q":
        # Generate multiple networks based on config file
        NOGRAPH = True

        operations = []
        with open(config, "r") as f:
            for line in f.readlines():
                if line.startswith("#"): continue
                if line.strip() == "": continue

                op = {}
                for pair in line.strip().split(" "):
                    key, value = pair.split(":")
                    op[key] = value
                operations.append(op)

        overallLoss = []

        for op in operations:
            start = time.time()
            print(f"Training {op['name']} (e: {op['epochs']}, s: {op['steps']}, r: {op['res']}, i: {op['input']}, h: {op['hidden']}, o: {op['output']}, l: {op['layers']}, d: {op['encdim']})")
            print(f"Start time: {time.ctime(start)}")

            model, res, loss = NeuralField.trainData(
                img=op['img'], name=op['name'], gens=int(op['epochs']), steps=int(op['steps']), res=int(op['res']),
                input_size=int(op['input']), hidden_size=int(op['hidden']), output_size=int(op['output']), hidden_layers=int(op['layers']), encdim=int(op['encdim'])
            )
            NeuralField.generateImage(res=res, model=model, name=f"{op['name']}/{op['name']}_{op['epochs']}", show=False, encdim=int(op['encdim']))
            toGif(op['name'])

            overallLoss.append(loss)

            end = time.time()
            print(f"End time: {time.ctime(end)}")
            print(f"Time elapsed: {end - start:.2f} seconds")
            print(f"Final loss: {loss}\n")

        plt.legend()
        plt.savefig(f"loss_{os.path.basename(config).split('.')[0]}.png")

        plt.figure()
        plt.plot(overallLoss)
        plt.xlabel("Dimension")
        plt.ylabel("Loss")
        plt.title("Loss Over Dimension")
        plt.savefig(f"loss_{os.path.basename(config).split('.')[0]}_dim.png")

        print("Losses:")
        print(", ".join(map(str, overallLoss)))
        print("All operations complete")
    else:
        error()
