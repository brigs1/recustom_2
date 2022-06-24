# coding=utf-8
import json
import os

import datasets

from PIL import Image
import numpy as np

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019recustom_2AD,
  title={recustom_2: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""
_DESCRIPTION = """\
https://github.com/brigs1/recustom_2/
"""

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

class recustom_2Config(datasets.BuilderConfig):
    """BuilderConfig for recustom_2"""

    def __init__(self, **kwargs):
        """BuilderConfig for recustom_2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(recustom_2Config, self).__init__(**kwargs)

class recustom_2(datasets.GeneratorBasedBuilder):
    """recustom_2 dataset."""

    BUILDER_CONFIGS = [
        recustom_2Config(name="recustom_2", version=datasets.Version("1.0.0"), description="recustom_2 dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/brigs1/recustom_2",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract("/home/y/Documents/Jupyter_tests/datasets/recustom_2/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            words = []
            bboxes = []
            ner_tags = []
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["form"]:
                words_example, label = item["words"], item["label"]
                words_example = [w for w in words_example if w["text"].strip() != ""]
                if len(words_example) == 0:
                    continue
                if label == "other":
                    for w in words_example:
                        words.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(w["box"], size))
                else:
                    words.append(words_example[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(words_example[0]["box"], size))
                    for w in words_example[1:]:
                        words.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(w["box"], size))
            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path}
