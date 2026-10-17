"""Compatibility entry point for shared FER preprocessing.

The historical mixed train/test CSV exporter is retired. Use a confirmed
dataset config and a fresh output directory; every split is exported separately.
"""

from open_clip_train.predata_fer import main


def generate_csv():
    main()


if __name__ == "__main__":
    generate_csv()
