import os
import random
import shutil


SOURCE = "dataset/train/1"

TRAIN = "dataset/train/1"

TEST = "dataset/test/1"


os.makedirs(
    TRAIN,
    exist_ok=True
)

os.makedirs(
    TEST,
    exist_ok=True
)


images = [
    file
    for file in os.listdir(SOURCE)
    if file.endswith(".png")
]


random.shuffle(images)


test_count = int(
    len(images) * 0.2
)

test_images = images[
    :test_count
]


for image in test_images:

    source = os.path.join(
        SOURCE,
        image
    )

    destination = os.path.join(
        TEST,
        image
    )

    shutil.move(
        source,
        destination
    )


print("Dataset split complete!")
print(f"Total: {len(images)}")
print(f"Train: {len(images) - test_count}")
print(f"Test : {test_count}")