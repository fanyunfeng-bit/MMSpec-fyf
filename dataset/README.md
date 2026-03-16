# MMSpec Dataset

**MMSpec** (Multimodal Speculative Decoding Benchmark) is a unified dataset for evaluating speculative decoding performance on vision-language models.

## Dataset Composition

| Subset | Source | Samples | Topic | Task Type |
|--------|--------|---------|-------|-----------|
| GQA Subset | [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/) | 100 | general vqa | Open-ended VQA |
| TextVQA Subset | [TextVQA](https://textvqa.org/) | 100 | text vqa | Text Reading VQA |
| COCO Captions | [COCO Dataset](https://cocodataset.org/) | 100 | image captioning | Image Captioning |
| CharXiv Subset | [CharXiv](https://charxiv.github.io/) | 100 | chart understanding | Chart Reasoning |
| MMMU_Pro Subset | [MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) | 100 | complex reasoning pro | Multiple-choice QA (10 options) |
| Multi-turn Subset | [ConvBench](https://huggingface.co/datasets/liushuo12345/ConvBench) + [MM-MT-Bench](https://huggingface.co/datasets/mistralai/MM-MT-Bench) | 100 | multi-turn conversation | Multi-turn VQA |
| **Total** | - | **600** | - | - |

## Directory Structure

```
dataset/
├── MMSpec/
│   ├── test/                     # Full test set
│   │   ├── mmspec.jsonl          # 600 records
│   │   └── images/               # 600 images (mmspec_001-600)
│   └── testmini/                 # Quick test subset
│       ├── mmspec.jsonl          # 60 records (10 per topic)
│       └── images/               # 60 images
└── README.md
```

## Data Format

```json
{
    "id": "mmspec_001",
    "image": "mmspec_001.jpg",
    "turns": ["Question text..."],
    "category": "default | Accounting | charts | ...",
    "topic": "general vqa | text vqa | image captioning | chart understanding | complex reasoning pro | multi-turn conversation"
}
```

## Reproducing the Dataset

### GQA Subset (100 samples, mmspec_001-100)

1. Download GQA testdev balanced split from [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/)
2. Load `llava_gqa_testdev_balanced.jsonl`
3. Shuffle with `Dataset.shuffle(seed=42)`
4. Select first 100 samples with **unique images**
5. Map to MMSpec format with IDs `mmspec_001` to `mmspec_100`, topic = `"general vqa"`

**Original Question IDs:**
```
202227935, 20899424, 201079761, 202003711, 20865555, 201185787, 202156890, 20903027, 
201068836, 20827259, 201996732, 201804660, 2075698, 201896554, 20491941, 201861265, 
20317285, 201794876, 201880477, 20794227, 202081092, 20480057, 202223187, 202286593, 
20434993, 20385650, 20541567, 20295524, 201065125, 201639249, 20178097, 201207117, 
2075356, 201462514, 201067634, 20818697, 20654994, 201400101, 202073332, 202147865, 
202108035, 201393537, 202122066, 201759403, 201972804, 201704599, 20412445, 201480278, 
201156113, 20588950, 20162320, 20416784, 201175530, 20898990, 20330200, 201322827, 
20709924, 201980742, 20248012, 201528095, 201590208, 202162013, 201428996, 201663614, 
201336960, 202266070, 201879053, 202218778, 20149593, 20341125, 20302785, 202125912, 
202144320, 20923055, 201654420, 20511576, 202243791, 20866501, 20711630, 202116873, 
20381224, 20644733, 20551620, 20518335, 20412198, 202060113, 201878353, 201641237, 
20754874, 201902497, 20287855, 201873237, 201407280, 201757653, 201997723, 20622082, 
20183113, 20600198, 201988059, 20896225
```

### TextVQA Subset (100 samples, mmspec_101-200)

1. Download TextVQA train split from [TextVQA](https://textvqa.org/)
2. Filter for samples where the **longest answer has >= 10 tokens**
3. Shuffle with `random.seed(42)`
4. Select first 100 samples with **unique images**
5. Map to MMSpec format with IDs `mmspec_101` to `mmspec_200`, topic = `"text vqa"`

**Filtering Criteria:**
- Answers must have >= 10 tokens (to ensure longer, more descriptive responses)
- Min/Max/Avg answer tokens: 10 / 69 / 14.3

**Original Question IDs:**
```
26593, 33647, 16354, 2143, 9259, 2458, 16634, 15443, 16806, 33391, 
26814, 9747, 21088, 26182, 26046, 5868, 4832, 783, 29699, 17266, 
25743, 26175, 16884, 25066, 4141, 1037, 17212, 4493, 10882, 22301, 
13299, 9248, 13253, 3465, 9064, 7934, 3468, 24850, 31147, 23991, 
3906, 16780, 28740, 8222, 32915, 16537, 1048, 17233, 2500, 6823, 
4419, 10083, 22584, 16870, 10791, 31132, 3187, 31864, 5129, 15451, 
26668, 30476, 3161, 22774, 3682, 1024, 16197, 18972, 33577, 1302, 
17090, 15763, 15334, 10922, 31170, 9450, 15203, 3136, 33087, 34584, 
4550, 26487, 29529, 18533, 6455, 4009, 34132, 25412, 7731, 19553, 
13499, 17133, 5025, 13808, 3164, 2186, 4686, 4414, 6032, 17037
```

### COCO Captions Subset (100 samples, mmspec_201-300)

1. Download COCO 2014 validation split from [COCO Dataset](https://cocodataset.org/)
2. Load `captions_val2014.json`
3. Shuffle image IDs with `random.seed(42)`
4. Select first 100 images
5. Use standard prompt: `"Describe this image in detail."`
6. Map to MMSpec format with IDs `mmspec_201` to `mmspec_300`, topic = `"image captioning"`

**Original Image IDs:**
```
15055, 503808, 530099, 541086, 75319, 6701, 104647, 264121, 291680, 431404, 
13965, 363071, 46099, 302242, 90738, 119414, 121417, 38029, 194832, 452013, 
260199, 234643, 206550, 202444, 180541, 126512, 146965, 537827, 183905, 83452, 
550013, 514607, 532055, 412471, 393647, 378831, 519329, 12896, 395210, 265872, 
499592, 564489, 195002, 154867, 205504, 521404, 198004, 203479, 126995, 74092, 
534605, 517523, 262262, 127192, 347747, 396350, 125997, 125286, 396274, 113082, 
318854, 47010, 91949, 300786, 290416, 410168, 480663, 350111, 125318, 374486, 
569535, 75426, 4275, 130613, 327624, 92115, 214200, 377012, 156100, 156567, 
428985, 109819, 267191, 505636, 529235, 88433, 76249, 451202, 324789, 298697, 
144379, 110282, 545730, 47131, 347142, 551737, 214574, 419265, 108904, 352584
```

### CharXiv Subset (100 samples, mmspec_301-400)

1. Download CharXiv validation split from [CharXiv Dataset](https://huggingface.co/datasets/princeton-nlp/CharXiv)
2. Load `val.parquet`
3. Shuffle with `random_state=42`
4. Select 100 samples
5. Use **reasoning questions** (more complex, require chart interpretation)
6. Map to MMSpec format with IDs `mmspec_301` to `mmspec_400`, topic = `"chart understanding"`

**Sample Questions:**
- Which chart shows the steepest increase for the AMPERE++ dataset across training set sizes?
- What is the name of the line that is the furthest away from its fi value from the W_H axis?
- At around what volume does the brown and blue curve intersect in the bottom left subplot?

**Original arXiv IDs:**
```
2003.14306, 2205.07656, 2212.06269, 2011.01739, 2010.13299, 2109.11156, 2307.11827, 2302.05247, 
2207.05939, 2007.09213, 2108.09750, 2209.14922, 2306.05565, 2306.07817, 2105.02488, 2001.04230, 
2012.01585, 2309.04116, 2305.09369, 2105.07618, 2010.01437, 2009.00003, 2107.14127, 2106.09353, 
2211.05288, 2203.14200, 2102.00108, 2306.08406, 2106.11810, 2106.03555, 2012.05430, 2310.07600, 
2101.01969, 2110.09915, 2310.11976, 2112.09814, 2109.02313, 2001.08012, 2107.00299, 2302.06966, 
2106.01191, 2012.03245, 2310.17239, 2302.01925, 2202.11041, 2201.01911, 2305.18343, 2302.03540, 
2312.01649, 2210.04690, 2109.03318, 2007.06038, 2103.03818, 2310.08605, 2106.08568, 2012.13975, 
2012.12207, 2302.14116, 2307.06019, 2306.12655, 2305.12933, 2007.07215, 2202.01362, 2308.08869, 
2005.04866, 2307.13158, 2006.14900, 2103.15060, 2109.03810, 2012.12243, 2310.12126, 2310.09754, 
2212.01482, 2210.11019, 2102.09368, 2304.00901, 2111.05077, 2001.08692, 2205.11384, 2310.09716, 
2212.10952, 2003.04603, 2006.11057, 2306.07195, 2003.12151, 2103.10935, 2305.19048, 2007.03331, 
2210.02411, 2009.10458, 2002.04488, 2310.17017, 2309.05076, 2109.12506, 2211.03067, 2010.07965, 
2305.05014, 2103.11728, 2305.17627, 2206.13540
```

### MMMU_Pro Subset (100 samples, mmspec_401-500)

1. Download MMMU_Pro (10 options) from [HuggingFace](https://huggingface.co/datasets/MMMU/MMMU_Pro)
2. Load test split from `standard (10 options)/test-*.parquet`
3. Filter for **single-image** samples only
4. Shuffle with `random_state=42`
5. Select 100 samples
6. Build prompts with question + 10 answer options
7. Map to MMSpec format with IDs `mmspec_401` to `mmspec_500`, topic = `"complex reasoning pro"`

**Sample Questions:**
- Which of the following best explains the overall trend shown in the image?
- What type of chemical bonding is represented in this molecular structure?

**Original MMMU_Pro IDs:**
```
test_Manage_191, test_History_12, test_Computer_Science_14, test_Mechanical_Engineering_382, 
test_Marketing_150, test_Pharmacy_355, validation_Psychology_5, validation_Computer_Science_2, 
validation_Energy_and_Power_19, test_Computer_Science_12, validation_Energy_and_Power_24, 
test_Sociology_68, test_Chemistry_30, validation_Physics_15, validation_Public_Health_7, 
validation_Mechanical_Engineering_10, validation_Economics_12, test_Pharmacy_415, 
test_Physics_369, validation_Physics_5...
```

### Multi-turn Subset (100 samples, mmspec_501-600)

This subset combines two multi-turn conversation benchmarks:

**ConvBench (47 samples, mmspec_501-547):**
1. Clone ConvBench from [HuggingFace](https://huggingface.co/datasets/liushuo12345/ConvBench)
2. Load `ConvBench.xlsx` and match with available images in `visit_bench_images/`
3. Extract 3-turn conversations (first, second, third turn instructions)
4. Categories: ~7 instruction categories

**MM-MT-Bench (53 samples, mmspec_548-600):**
1. Clone MM-MT-Bench from [HuggingFace](https://huggingface.co/datasets/mistralai/MM-MT-Bench)
2. Load eval split parquet file
3. Priority: samples with >1 turns first, then longest single-turn samples
4. Sample 53 records to complete 100 multi-turn samples

**Multi-turn Format:**
Unlike other subsets with single-turn questions, multi-turn samples have multiple `turns`:
```json
{
    "turns": ["First question...", "Second question...", "Third question..."]
}
```

## License

GQA, TextVQA, COCO, CharXiv, MMMU_Pro, ConvBench, MM-MT-Bench datasets are subject to their respective licenses.
