import sys, os
from PIL import Image


def toGif(inputPath: str):
    frames: list[Image.Image] = []

    sortLambda = lambda x: int(x.split(".")[0].split("_")[-1])
    names = sorted([file for file in os.listdir(inputPath) if file.endswith(".png") and not file.startswith("base") and not file.startswith("loss")], key=sortLambda)

    for file in names:
        frames.append(Image.open(os.path.join(inputPath, file)))

    if len(frames) == 0:
        print(f"No PNG files found in {inputPath}")
        return
    
    print(f"Creating GIF from {os.path.basename(inputPath)} -> output\\{os.path.basename(inputPath)}.gif")

    frames[0].save(
        fp=f"{inputPath}\\{os.path.basename(inputPath)}.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=250,
        loop=0,
        disposal=0,
    )

    for file in names:
        if file.startswith("gen_"):
            os.remove(os.path.join(inputPath, file))


def main():
    if len(sys.argv) < 2:
        print("Usage: python toGif.py <input_path>")
        sys.exit(1)

    inputPath = sys.argv[1]

    if not os.path.exists(inputPath):
        print("The input path does not exist!")
        sys.exit(1)

    if os.path.isdir(inputPath):
        toGif(inputPath)
    else:
        print("Please provide a directory as input path!")


if __name__ == "__main__":
    main()