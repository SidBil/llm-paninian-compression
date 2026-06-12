# LLM Pāṇinian Compression

Research project training a character-level decoder-only Transformer on SLP1-transliterated Sanskrit, then probing its internal activations for evidence that the model has rediscovered **phonological and morphophonological structure** (Pāṇinian patterns) without being explicitly taught them.

The core claim: a neural network trained purely on next-character prediction over Sanskrit text internally organises representations in ways that parallel Pāṇini's grammatical categories — vowel length, place of articulation, sandhi junctions, morpheme boundaries.

---

## Status

| Phase | Status |
|---|---|
| Corpus pipeline | ✅ Complete — 1,767,411 clean SLP1 lines |
| Model training | ✅ Complete — val loss 1.03 nats (1.49 BPC) at 50k iters |
| Activation probing | 🔄 In progress |

---

## Repository Structure

```
llm-paninian-compression/
├── corpus/
│   ├── requirements.txt
│   └── src/
│       ├── 01_acquire.ipynb        # Download GRETIL + DCS sources
│       ├── 02_extract.ipynb        # Extract lines; pre-transliteration noise stripping
│       ├── 03_normalize.ipynb      # IAST→SLP1; whitelist cleaning
│       ├── 04_deduplicate.ipynb    # Deduplication
│       └── 05_report.ipynb         # Corpus stats + validation
├── training/
│   ├── sanskrit_lm_feasibility.ipynb   # Feasibility model: 4L×4H×256D, ~3M params (Colab)
│   └── sanskrit_lm_v1.ipynb            # V1 model: 6L×8H×512D, ~19M params (Colab)
└── probing/
    └── sanskrit_probe_v1.ipynb         # Probing dataset builder: DCS → activations (Colab)
```

Data (`corpus/data/`) is excluded from version control — see corpus pipeline setup below.

---

## Model

`SanskritLM` is a decoder-only Transformer:

- **Architecture**: 6 layers × 8 heads × 512D, ~19M parameters
- **Vocabulary**: ~45–50 characters, built dynamically from corpus (no hardcoded alphabet)
- **Context window**: 512 characters
- **Input**: flat SLP1 character sequence; newline `\n` is a vocabulary token
- **Training objective**: next-character cross-entropy over random 512-char windows
- **Key features**: flash attention, weight tying (embedding = output head), pre-norm, fp16

Trained on Google Colab (T4 GPU) for 50,000 iterations. Checkpoints saved to `MyDrive/sanskrit/v1_run/`.

---

## SLP1

SLP1 is an ASCII-based Sanskrit transliteration scheme. Every phoneme is one character — no digraphs.

| SLP1 | IPA / IAST |
|---|---|
| `A I U` | ā ī ū (long vowels) |
| `E O` | e o |
| `M H` | ṃ (anusvāra) ḥ (visarga) |
| `S` | ś (palatal sibilant) |
| `z` | ṣ (retroflex sibilant) |
| `B D G` | bh dh gh (voiced aspirates) |
| `f` | ṛ |
| `w q` | ṭ ḍ (retroflexes) |

The `.` in corpus lines is the daṇḍa (Sanskrit clause punctuation), not noise.

---

## Corpus Pipeline

The pipeline runs locally. Data is not committed to the repo (GRETIL + DCS raw files are large).

### Setup

```bash
cd corpus
source venv/bin/activate
```

### Run a single step

```bash
./venv/bin/python3.14 -m nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=7200 \
  --output src/executed/03_normalize.ipynb \
  src/03_normalize.ipynb
```

Run notebooks 01 → 05 in order. Outputs land in `src/executed/`.

### Final corpus

`data/clean/corpus.slp1.txt` — 1,767,411 lines, ~175M characters, 0 bad characters.

---

## Probing

`sanskrit_probe_v1.ipynb` runs on Colab and:

1. Parses DCS CoNLL-U files → SLP1 sentences with character-level span offsets per case-bearing nominal
2. Counts per-class case labels (hard checkpoint before any GPU time)
3. Loads the trained `SanskritLM` checkpoint
4. Extracts contextual activations at all 6 layers (`final_char` + `mean_pool` variants)
5. Saves `probing_activations.npz` + `probing_metadata.csv` to Drive

Probing targets: do layer activations cluster by vowel length, sibilant type, place of articulation? Is there a discontinuity at sandhi junctions vs. word-internal positions?

---

## Data Sources

- **GRETIL**: ~781 Sanskrit plaintext files, CC BY-NC-SA 4.0
- **DCS** (Oliver Hellwig): ~15,900 CoNLL-U files with morphological annotation, CC BY 4.0
