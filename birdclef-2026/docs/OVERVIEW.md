# BirdCLEF+ 2026 — Competition Overview

- **Competition page:** https://www.kaggle.com/competitions/birdclef-2026
- **Sponsor:** Google Research & Cornell Lab of Ornithology
- **Category:** Research
- **Total prize pool:** $50,000 (+ $5,000 working-note awards)
- **Data license:** CC BY-NC-SA 4.0
- **Winner license:** Open-source (OSI-approved)

## Abstract

The goal of this competition is to develop machine learning frameworks capable of
identifying understudied species within continuous audio data from Brazil's
Pantanal wetlands. Successful solutions will help advance biodiversity monitoring
in the last wild places on Earth.

## Task

Identify which species (birds, amphibians, mammals, reptiles, insects) are
calling in recordings made in the **Brazilian Pantanal**. This is an important
task for scientists who monitor animal populations for conservation purposes.
More accurate solutions could enable more comprehensive monitoring.

The competition uses a **hidden test set**: when the submitted notebook is
scored, the actual test data is mounted into the notebook environment at runtime.

## Timeline

| Date | Milestone |
| --- | --- |
| 2026-03-11 | Start date |
| 2026-05-27 | Entry deadline (must accept rules by this date) |
| 2026-05-27 | Team merger deadline |
| 2026-06-03 | **Final submission deadline** |
| 2026-06-17 | Working note paper submission deadline (CLEF 2026, optional) |
| 2026-06-24 | Working note notification of acceptance |
| 2026-07-06 | Working note camera-ready deadline |

All deadlines are 23:59 UTC.

## Evaluation

Metric: **macro-averaged ROC-AUC** that skips classes with no true positive labels.
See https://www.kaggle.com/code/metric/birdclef-roc-auc.

### Submission Format

For each `row_id`, predict the probability that each species was present.
There is one column per species (234 species columns).
**Each row covers a five-second window of audio.**

`row_id` format: `[soundscape_filename]_[end_time]`,
e.g. segment 00:15–00:20 of `BC2026_Test_0001_S05_20250227_010002.ogg` has
`row_id` = `BC2026_Test_0001_S05_20250227_010002_20`.

## Code Requirements

**This is a Code Competition.** Submissions must be made through Kaggle Notebooks.
For the "Submit" button to be active after a commit, the following conditions must be met:

- CPU notebook **≤ 90 minutes** run-time
- **GPU notebook submissions are disabled** (technically submittable, but only 1 minute runtime — effectively unusable)
- **Internet access disabled** during submission
- Freely & publicly available external data is allowed, **including pre-trained models**
- Submission file must be named `submission.csv`

See the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ)
and the [code debugging doc](https://www.kaggle.com/code-competition-debugging).

## Prizes

| Place | Prize |
| --- | --- |
| 1st | $15,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $7,000 |
| 5th | $5,000 |

### Best Working Note Award (optional)

Participants are encouraged to submit a working note to the
[CLEF 2026 conference](https://clef2026.clef-initiative.eu/).
The top two notes each receive **$2,500**.

Judging criteria (max 15 pts, 2 reviewers averaged):

- **Work & contribution** (1–5): excellent contribution → meets scientific standards
- **Originality & novelty** (1–5): trailblazing → "been said many times before"
- **Readability & organization** (1–5): excellent → needs considerable work

## Data Description

### `train_audio/`
Short recordings of individual bird, amphibian, reptile, mammal and insect sounds
from [xeno-canto.org](https://www.xeno-canto.org/) and
[iNaturalist](https://www.inaturalist.org). Resampled to **32 kHz**, `.ogg` format.
Filenames: `[collection][file_id_in_collection].ogg`.

The training data is intended to contain nearly all relevant files; the organizers
ask participants **not** to additionally scrape xeno-canto or iNaturalist.

### `test_soundscapes/`
Populated at submission time with **~600** recordings, **1 minute** each, `.ogg` @ 32 kHz.
Filenames: `BC2026_Test_<file ID>_<site>_<date>_<time UTC>.ogg`
(e.g. `BC2026_Test_0001_S05_20250227_010002.ogg`).
It takes approximately **5 minutes** to load all test soundscapes.

**Not all species from the training data actually occur in the test data.**

### `train_soundscapes/`
Additional audio from roughly the same recording locations as the test soundscapes.
Same filename convention. Recording sites can overlap with test, but **dates/times
do not overlap**.

This year a subset is **labeled by expert annotators**. Ground truth is in
`train_soundscapes_labels.csv` with columns:

- `filename` — soundscape filename
- `start`, `end` — 5-second segment (as `HH:MM:SS` strings)
- `primary_label` — **semicolon-separated** species codes present in segment (multi-label)

**Important:** Some species in the hidden test data may appear in training **only**
inside the labeled `train_soundscapes` (not in `train_audio`). However, not all
species in `train_soundscapes` appear in `test_soundscapes`.

### `train.csv`
Metadata for `train_audio`. Key columns:

- `primary_label` — species code (eBird code for birds, iNaturalist taxon ID for non-birds).
  e.g. https://ebird.org/species/brnowl, https://www.inaturalist.org/taxa/41970
- `secondary_labels` — list of additional species marked as present (may be incomplete)
- `latitude`, `longitude` — recording coordinates (some species have local "dialects")
- `author` — uploader, or `Unknown`
- `filename` — audio filename
- `rating` — 1–5 (xeno-canto quality; −0.5 if background species present; 0 = unrated; 0 for all iNat)
- `collection` — `XC` (xeno-canto) or `iNat` (iNaturalist)
- Plus `type`, `scientific_name`, `common_name`, `class_name`, `inat_taxon_id`, `license`, `url`

### `sample_submission.csv`
Valid sample submission with 234 species ID columns.

### `taxonomy.csv`
Data on the 234 classes including iNaturalist taxon ID and `class_name`
(`Aves`, `Amphibia`, `Mammalia`, `Insecta`, `Reptilia`).

**Most insect species are not identified to species level** — they appear as
sonotypes (e.g. `47158son16` = insect sonotype 16). These sonotypes are treated
as classes despite the lack of species ID, and some occur in test data.
The 234 rows correspond to the 234 class columns in the submission file.
`primary_label` specifies the submission column name.

### `recording_location.txt`
> Pantanal, Mato Grosso do Sul, Brazil, South America
> More info: https://en.wikipedia.org/wiki/Pantanal
>
> Coordinates of recorder deployment sites:
> Latitude: -16.5 to -21.6
> Longitude: -55.9 to -57.6

## Dataset Size

- **Total: ~16.1 GB**
- 46,207 `.ogg` audio files (~16.13 GB)
- 3 CSV files actually downloadable (~6.8 MB) — `train.csv`, `taxonomy.csv`, `sample_submission.csv`
  (API file_summary reports 4 CSVs, but `train_soundscapes_labels.csv` is not yet released)
- 2 TXT files (~282 B) — `recording_location.txt`, `test_soundscapes/readme.txt`

### Local layout (under `birdclef-2026/data/`, gitignored)

```
data/
├── recording_location.txt
├── sample_submission.csv
├── taxonomy.csv
├── train.csv                       (35,549 rows)
├── test_soundscapes/readme.txt     (test files appear only at submission time)
├── train_audio/         (35,549 .ogg across 206 species dirs, ~11 GB)
└── train_soundscapes/   (~10,644 .ogg, ~5.1 GB; date-coded BC2026_Train_*.ogg)
```

> 14 of 10,658 `train_soundscapes/` files (~7 MB) couldn't be re-downloaded due
> to Kaggle 429 rate-limit. List: `birdclef-2026/missing_files.txt`. Run
> `kaggle competitions download -c birdclef-2026 -f <path>` later to backfill.

### Class distribution (234 target classes)

| Class | # species |
| --- | ---: |
| Aves       | 162 |
| Amphibia   |  35 |
| Insecta    |  28 |
| Mammalia   |   8 |
| Reptilia   |   1 |

### `train.csv` distribution (35,549 rows)

| Class | # clips |
| --- | ---: |
| Aves      | 34,799 |
| Amphibia  |    451 |
| Insecta   |    199 |
| Mammalia  |     99 |
| Reptilia  |      1 |

Source split: **23,043 from xeno-canto**, **12,506 from iNaturalist**.

> **Heads up:** `train.csv` contains only **206 unique primary labels**, but the
> taxonomy has 234 classes. That means **28 target species have no `train_audio`
> examples** — they only appear via the labeled portion of `train_soundscapes`
> (`train_soundscapes_labels.csv`). Plan your training/validation around this.

## Acknowledgements

Dataset development supported by the
[Bezos Earth Fund AI for Climate and Nature Grand Challenge](https://www.bezosearthfund.org/news-and-insights/bezos-earth-fund-announces-30-million-in-ai-grand-challenge-awards).

Contributing institutions (alphabetical):

- **Chemnitz University of Technology** — Stefan Kahl, Mario Lasseck, Maximilian Eibl
- **Google DeepMind** — Tom Denton
- **iNaturalist** — Grant van Horn
- **Instituto Homem Pantaneiro** — Wener Hugo Arruda Moreno
- **Instituto Nacional de Pesquisa do Pantanal (INPP)** — Carolline Zatta Fieker, Karl-L. Schuchmann, Kirk Thiago Pedroso Azevedo, Lucas Korzune Sampaio Teles, Marinez Isaac Marques, Matheus Gonçalves dos Reis
- **K. Lisa Yang Center for Conservation Bioacoustics** — Stefan Kahl, Larissa Sugai, Holger Klinck
- **LifeCLEF** — Alexis Joly, Henning Müller
- **Sauá Consultoria Ambiental** — Carolina Martins Garcia
- **Universidade Federal de Mato Grosso do Sul (UFMS)** — Alyson Vieira de Melo, Daiene Louveira Hokama Sousa, José Luiz Massao Moreira Sugai, João Emílio de Almeida Júnior, Liliana Piatti, Mariana Motti Barbosa, Matheus de Oliveira Neves, Priscila do Nascimento Lopes, Ryan Christopher Kridler
- **Xeno-canto** — Willem-Pier Vellinga, Bob Planqué

Photo credits: Hyacinth Macaw banner by Thomas Fuhrmann; Jaguar inset by Leonardo Ramos.
