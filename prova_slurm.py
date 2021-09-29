import os
import torch


def main():
    path = os.path.join("file_prova")
    if not os.path.exists(path):
        os.makedirs(path)
    fp = open(os.path.join(path, "prova.txt"), "w")
    fp.write("ciao\n")
    print("Dovrebbe andare nel file di output")
    fp.write(str(10))
    print("--- Prova loading del modello ---")
    model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    model = model.to('cuda')
    fp.write("Model loaded in GPU")
    fp.close()


if __name__ == '__main__':
    main()
