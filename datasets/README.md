# Data

This file documents only dataset paths under `csteer/datasets/`.

## Required local layout

```text
datasets
├── ViP-Bench
│   ├── vip-bench-meta-data.json
│   ├── bbox
│   │   ├── questions.jsonl
│   │   └── images
│   │       ├── *.png
│   │       └── ...
│   └── source_image                # optional for noref mode
│       ├── *.png
│       └── ...
├── Inst-It-Bench
│   ├── image_multi_choices.json
│   ├── image_open_ended.json
│   ├── image_instance_captions_masks.json
│   ├── video_multi_choices.json
│   ├── video_open_ended.json
│   ├── video_instance_captions_masks.json
│   ├── images_vpt                  # refer images
│   │   ├── 001.jpg
│   │   └── ...
│   ├── images_raw                  # noref images
│   │   ├── 001.jpg
│   │   └── ...
│   ├── videos_vpt                  # refer frame folders
│   │   ├── <video_path>
│   │   │   ├── *.jpg
│   │   │   └── ...
│   │   └── ...
│   └── videos_raw                  # noref frame folders
│       ├── <video_path>
│       │   ├── *.jpg
│       │   └── ...
│       └── ...
└── GAR
    ├── GAR-Bench-VQA.json
    ├── GAR-Bench-Caption-Simple.json
    ├── GAR-Bench-Caption-Detailed.json
    ├── vqa-images
    │   ├── *.png
    │   └── ...
    ├── simple-images
    │   ├── *.png
    │   └── ...
    ├── detailed-images             # refer detailed images
    │   ├── *.png
    │   └── ...
    └── images                      # noref detailed images
        ├── *.png
        └── ...
```

## Notes

- ViP-Bench eval uses `bbox/questions.jsonl` and `vip-bench-meta-data.json`.
- Inst-It eval scripts may load GT from HuggingFace (`Inst-IT/Inst-It-Bench`) by default, but local JSON/assets are still used by loaders.
- BLINK and CV-Bench are currently loaded from ModelScope (`evalscope/BLINK`, `comefly/cvbench`), so they do not require local folders under `datasets/` unless you build an offline mirror.
