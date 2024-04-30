# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datasets
from datasets.tasks import ImageClassification
from collections import Counter
import tarfile
from imagenet_classes import IMAGENET2012_CLASSES

_CITATION = """\
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
"""

_HOMEPAGE = "https://image-net.org/index.php"

_DESCRIPTION = """\
ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, ImageNet hopes to offer tens of millions of cleanly sorted images for most of the concepts in the WordNet hierarchy. ImageNet 2012 is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images.
"""

_DATA_URL = {
    "train": [f"Data/train_images_{i}.tar.gz" for i in range(5)],
    "val": ["Data/val_images.tar.gz"],
    "test": ["Data/test_images.tar.gz"],
}

class Imagenet1k(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        assert len(IMAGENET2012_CLASSES) == 1000
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=list(IMAGENET2012_CLASSES.values())),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[ImageClassification(image_column="image", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        archives = dl_manager.download(_DATA_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archives": [dl_manager.iter_archive(archive) for archive in archives["train"]],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "archives": [dl_manager.iter_archive(archive) for archive in archives["val"]],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "archives": [dl_manager.iter_archive(archive) for archive in archives["test"]],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, archives, split):
        """Yields examples and counts images per class using the IMAGENET2012_CLASSES for descriptive labeling."""
        idx = 0
        class_counts = {name: 0 for name in IMAGENET2012_CLASSES.values()}  # Initialize counts for each descriptive class name

        for archive in archives:
            for path, file in archive:
                if path.endswith(".JPEG"):
                    if split != "test":
                        root, _ = os.path.splitext(path)
                        _, synset_id = os.path.basename(root).rsplit("_", 1)
                        label = IMAGENET2012_CLASSES[synset_id]
                    else:
                        label = None  # Use None or similar for test images if they should not be counted

                    if label is not None:  # Count only if label is known
                        class_counts[label] += 1

                    ex = {"image": {"path": path, "bytes": file.read()}, "label": label}
                    yield idx, ex
                    idx += 1

        # Optional: Print the image counts per class
        print("Image counts per class:", class_counts)


    def count_images_in_archives(self):
        # This will hold the count of images per class
        class_counts = Counter()
        
        # Define the base path for the Data directory
        base_path = '../Data'
        
        # Tuples of (filename pattern, number of files) for each split
        archive_patterns = [
            ('train_images_{}.tar.gz', 5),
            ('val_images.tar.gz', 1),
            ('test_images.tar.gz', 1)
        ]
        
        for pattern, count in archive_patterns:
            for i in range(count):
                # Adjust the file path template for train images
                if '{}' in pattern:
                    file_path = os.path.join(base_path, pattern.format(i))
                else:
                    file_path = os.path.join(base_path, pattern)
                
                # Open the tar file and count images
                with tarfile.open(file_path, 'r:gz') as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.endswith('.JPEG'):
                            # Extract the synset ID from the file name
                            synset_id = os.path.basename(member.name).split('_')[0]
                            label = IMAGENET2012_CLASSES.get(synset_id)
                            if label:
                                class_counts[label] += 1
        
        return class_counts
    
