import os
import filetype
import shutil

for dataset in os.listdir("generated_images"):
    for run in os.listdir("generated_images/{}".format(dataset)):
        run_path = "generated_images/{}/{}".format(dataset, run)
        found_png = False
        for element in os.scandir(run_path):
            if element.is_dir():
                continue
            kind = filetype.guess(element.path)
            if kind is not None and kind.mime == "image/png":
                found_png = True
                break

        if not found_png:
            print("no images in {}".format(run_path))
            shutil.rmtree(run_path)
