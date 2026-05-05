## vip_bench

#### ../DATA/ViP-Bench/vip-bench-meta-data.json
```
    {
        "v1_0": {
            "image_source": "v1_1.png",
            "image": "v1_1_0.png",
            "question": "What is the value of the variable in the equation <obj>?",
            "answer": "0.75<OR>x=0.75<OR>3/4",
            "capability": [
                "ocr",
                "math"
            ]
        },
        ...
    }
    
```

#### source_image_dir
```
    ../DATA/ViP-Bench/source_image
    | - v1_1.png
    ...
```

## gar_detail_oe

#### no refer
```
    ../DATA/GAR/images
    | - caption_detailed_64.png
    | ...
```

#### refer
```
    ../DATA/GAR/detailed-images
    | - caption_detailed_64.png
    | ...
```

## inst_it_image_mc/oe

#### what refer looks like now
```
    from datasets import load_dataset
    ds = load_dataset("../DATA/Inst-It-Bench", "image_multi_choice"/"image_open_ended", split="test")
```

#### ../DATA/Inst-It-Bench/image_multi_choices.json
```
    {
        "image_id": 1,
        "question_id": "001-01",
        "image_path": "images_vpt/001.jpg",
        "question": "What is [3] carrying in their hand?",
        "answer": "C",
        "options": {
            "A": "[3] is carrying a handbag in their hand.",
            "B": "bag",
            "C": "[3] is holding [4], which appears to be a shopping bag.",
            "D": "[3] is holding [5], which appears to be a laptop case."
        },
        "meta_info": {
            "dataset": "BRUST",
            "split": "val"
        }
    },
    ...
```

#### ../DATA/Inst-It-Bench/image_open_ended.json
```
    {
        "image_id": 1,
        "question_id": "001-01",
        "image_path": "images_vpt/001.jpg",
        "question": "What is [3] carrying in their hand?",
        "answer": "[3] is holding [4], which appears to be a shopping bag.",
        "meta_info": {
            "dataset": "BRUST",
            "split": "val"
        }
    },
    ...
```

#### no refer
```
    ../DATA/Inst-It-Bench
    | - images_raw
        | - 001.jpg
        | - ...
```

#### refer
```
    ../DATA/Inst-It-Bench
    | - images_vpt
        | - 001.jpg
        | - ...
```

