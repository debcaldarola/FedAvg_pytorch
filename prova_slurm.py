import os


def main():
    path = os.path.join("file_prova")
    if not os.path.exists(path):
        os.makedirs(path)
    fp = open(os.path.join(path, "prova.txt"), "w")
    fp.write("ciao\n")
    fp.write(str(10))
    fp.close()


if __name__ == '__main__':
    main()
