# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project training a character-level decoder-only Transformer on SLP1-transliterated Sanskrit, with the goal of extracting internal activations to probe for phonological structure (Pāṇinian grammar patterns).

Two phases:
1. **Corpus pipeline** (`corpus/`) — builds a clean SLP1 corpus from GRETIL + DCS sources, runs locally
2. **LM training** (`training/sanskrit_lm_feasibility.ipynb`, `training/sanskrit_lm_v1.ipynb`) — trains and evaluates the model, designed for Google Colab with a T4 GPU

**Current status**: Both phases are complete. The model is trained (50k iters, val loss 1.03 nats / 1.49 BPC). Next work is activation probing — see [Probing Experiments](#probing-experiments-next-steps) below.

## SLP1 Format

SLP1 is an ASCII-based Sanskrit transliteration scheme. Uppercase letters denote special phonemes: `A`=ā, `I`=ī, `U`=ū, `E`=e, `O`=o, `B`=bh, `S`=ś, `M`=ṃ, `H`=ḥ, `f`=ṛ, `z`=ṣ, `w`=ṭ, `q`=ḍ.

The corpus whitelist allows only: `A-Za-z`, space, `'` (avagraha), `.` (daṇḍa — Sanskrit clause punctuation, **not** noise), `|` (daṇḍa alt). All other characters are stripped by `clean_slp1()` in `corpus/src/03_normalize.ipynb` (cell `0f808b2d`).

## Corpus Pipeline

### Setup

```bash
cd corpus
source venv/bin/activate
pip install -r requirements.txt   # if reinstalling
```

### Run a single notebook step

```bash
cd corpus
./venv/bin/python3.14 -m nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=7200 \
  --output src/executed/03_normalize.ipynb \
  src/03_normalize.ipynb
```

### Pipeline stages

| Notebook | Input | Output |
|---|---|---|
| `src/01_acquire.ipynb` | — | `data/raw/gretil/`, `data/raw/dcs/`, `PROVENANCE.txt` |
| `src/02_extract.ipynb` | raw files | `data/interim/gretil_extracted.jsonl`, `dcs_extracted.jsonl` |
| `src/03_normalize.ipynb` | extracted JSONL | `data/interim/corpus_slp1.jsonl` |
| `src/04_deduplicate.ipynb` | `corpus_slp1.jsonl` | `data/clean/corpus.slp1.txt`, `data/clean/index.csv` |
| `src/05_report.ipynb` | clean corpus | stats, validation |

Executed output lands in `src/executed/`.

### Final corpus state

`data/clean/corpus.slp1.txt` — **1,767,411 lines**, ~175M characters, verified 0 bad characters. `data/clean/index.csv` maps each line to `text_id` (needed for future work-level train/val splits).

## LM Notebooks (Google Colab)

Both notebooks are self-contained and designed for Colab + T4 GPU. They are **not** meant to run locally.

### Workflow

1. Upload `corpus/data/clean/corpus.slp1.txt` to `MyDrive/sanskrit/corpus.slp1.txt`
2. **CRITICAL before fresh training**: delete any existing `checkpoint.pt` and `best_checkpoint.pt` from the Drive output directory — the notebooks auto-resume from them
3. Open the notebook in Colab, run cells top-to-bottom
4. On disconnect, re-run Cells 0–4 to resume from the last checkpoint automatically

### Feasibility notebook (`training/sanskrit_lm_feasibility.ipynb`)

- Model: 4L × 4H × 256D, ~3M params, context 256 tokens
- 3,000 training iters (~15–20 min on T4)
- Output dir: `MyDrive/sanskrit/feasibility_run/`
- Four checks: loss curve, generation sample, activation extraction, compute report

### V1 notebook (`training/sanskrit_lm_v1.ipynb`) — TRAINED

- Model: 6L × 8H × 512D, ~19M params, context 512 tokens, flash attention, weight tying
- **Trained for 50,000 iters; final val loss ~1.03 nats (1.49 BPC), train loss ~1.116**
- Val loss < train loss throughout — expected: the val split (tail of corpus) is more regular/formulaic Sanskrit
- Cosine LR decay with linear warmup; saves both rolling `checkpoint.pt` and `best_checkpoint.pt`
- Output dir: `MyDrive/sanskrit/v1_run/`
- Keep-alive cell (Cell 1a) prevents Colab from disconnecting — run it first

### Model architecture (both notebooks)

`SanskritLM` is a decoder-only Transformer with:
- Token + positional embeddings → N causal self-attention blocks → LayerNorm → linear head
- Pre-norm (LN before attention and MLP)
- Vocabulary: ~45–50 characters, built dynamically from corpus (no hardcoded alphabet); `\n` is a vocabulary token
- `forward(idx, targets=None, return_hidden_states=False)` — when `return_hidden_states=True`, returns a list of `(B, T, n_embd)` residual-stream tensors after each block, used for phoneme-boundary probing

## Probing Experiments (Next Steps)

The model is the probe target, not the product. The goal is showing activations encode Pāṇinian structure without being explicitly taught it.

**Probing targets** (in priority order):
- Do activations cluster by phonological features (vowel length, place of articulation, sibilant type)?
- Is there a discontinuity at sandhi junctions vs. word-internal positions?
- Do layers specialise at different levels (phoneme → syllable → morpheme)?

**Implementation plan**:
1. `probing/sanskrit_probe_v1.ipynb` — exists, ready to run on Colab
2. Load `best_checkpoint.pt` from Drive using Cell 6 of `training/sanskrit_lm_v1.ipynb` as template
3. Design probe sentences with known character-level labels. Examples:
   - Sandhi: `tatrEka` (position 5 = sandhi of `a+e=E`)
   - Vowel length: `kavi` (short i) vs `kavI` (long I)
   - Sibilant: `sa` (dental) vs `Sa` (palatal) vs `za` (retroflex)
4. Call `model.forward(idx, return_hidden_states=True)` — Cell 7 of `sanskrit_lm_v1.ipynb` saves the 6 layer tensors (shape `(1, T, 512)`) to Drive
5. Train linear probes (logistic regression per layer) to predict each phonological label
6. PCA/UMAP on activation vectors coloured by category for visual evidence

**Important**: For probing, use custom hand-annotated sentences — not the val set. You need character-level ground-truth labels that don't exist for raw corpus text.

## Key Design Decisions

- **Whitelist-first cleaning**: `clean_slp1()` applies `_WHITELIST_RE = re.compile(r"[^A-Za-z .'\|]")` inside the segment loop — this single pass eliminates all noise. Iterative per-pattern patching failed to converge (6+ iterations, still 15% noisy lines).
- **Pre-transliteration noise stripping**: Mahābhāṣya cross-reference codes (`(pas_1)`, `ka_i,1.1-5`, `{1/10}`) must be stripped in `corpus/src/02_extract.ipynb` (cell `864c3b91`) **before** IAST→SLP1 conversion — their Latin letters convert to valid SLP1 characters and become indistinguishable downstream.
- **Three-pass trailing cleanup**: `_TRAIL_NUMS` → `_TRAIL_HYPHEN` → `_TRAIL_NUMS` handles verse refs like `55 - 70` where a hyphen hides trailing numbers from a single scan.
- **Hyphens stripped**: Hyphens are editorial in GRETIL (e.g., `gopa-jana` → `gopajana`). Stripping them is correct for SLP1.
- **`_ABBREV_RUN`** (`^(?:[a-z]\.\s+)+`): strips leading grammar/prosody abbreviation prefixes (`p. `, `p. r. y. `) from commentary texts in normalization.
- **Train/val split is contiguous 90/10**: Acceptable for feasibility and V1. Production probing experiments should use work-level splits keyed on `text_id` from `data/clean/index.csv` to prevent leakage.

## Data Sources

- **GRETIL**: ~781 Sanskrit plaintext files (IAST), CC BY-NC-SA 4.0
- **DCS**: ~15,790 CoNLL-U files (IAST), CC BY 4.0 (Oliver Hellwig)
- Both converted to SLP1 via `indic-transliteration`; round-trip check (SLP1 → original → SLP1) must pass < 0.5% mismatch rate before writing output

## Known Issues

- **German editorial lines**: ~166 lines from ~15 GRETIL files contain German prose (0.009% of corpus). No reliable filter without false-positives on short-vowel Sanskrit — left in corpus, negligible for training.
- **Repetition loops in generation**: autoregressive collapse when a high-frequency word gets reinforced at the end of long generations. Fix at inference time: higher temperature or repetition penalty.
