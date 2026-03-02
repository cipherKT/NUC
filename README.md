# NUC Algorithms for 16-bit Image Sequences

This repository provides implementations of multiple Scene-Based Non-Uniformity Correction (NUC) algorithms for 16-bit frame sequences (for example, thermal/IR data), with a unified CLI and optional automatic hyperparameter search.

## Implemented Methods

- `twopointnuc`: Scene-Based Two-Point NUC
- `multisegmentnuc`: Multi-Segment NUC
- `cssbnuc`: Constant-Statistics Scene-Based NUC
- `lmssbnuc`: LMS Scene-Based NUC variants
  - `standard`
  - `adaptive`
  - `gated`

## Repository Structure

```text
.
|-- main.py
|-- requirements.txt
|-- algorithms/
|-- optimal/
|-- data/
|-- utils/
`-- agent_files/
```

- `algorithms/`: Core algorithm implementations.
- `optimal/`: Hyperparameter optimization helpers.
- `data/datasetHandler.py`: Frame loading/saving.
- `utils/metrics.py`: `col_mad`, `row_mad`, `spatial_mad`, `rnu`.
- `main.py`: CLI entrypoint and dispatch.

## 1. Clone

```bash
git clone https://github.com/cipherKT/NUC.git
cd implementations
```

## 2. Create Environment and Install Dependencies

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Dataset Format

- Input must be a directory of **16-bit PNG** frames.
- All frames must have the same shape.
- Example:

```text
data/raw_seq/
|-- frame_0000.png
|-- frame_0001.png
|-- frame_0002.png
`-- ...
```

## 4. Basic Usage

```bash
python main.py -d <raw_frames_dir> -o <output_dir> -m <method>
```

Required flags:
- `-d, --data`: input frame directory
- `-o, --output`: output directory for corrected frames
- `-m, --method`: one of `twopointnuc`, `multisegmentnuc`, `cssbnuc`, `lmssbnuc`

Optional:
- `-s, --show_frames`: visualize raw vs corrected frames

## 5. Hyperparameter Optimization Mode

Optimization is **enabled by default** for supported methods.

- Disable optimization:
  - `--no-optimize`
- Use smaller search grids:
  - `--optimize-fast`

### Custom Candidate Lists (comma-separated)

- Two-Point:
  - `--opt-twopoint-ratios`
  - `--opt-twopoint-regions`
- Multi-Segment:
  - `--opt-multisegment-regions`
- Constant-Statistics:
  - `--opt-css-alpha`
  - `--opt-css-T`
- LMS:
  - `--opt-lms-epsilon`
  - `--opt-lms-K`
  - `--opt-lms-M-scale`
  - `--opt-lms-T`

Example:

```bash
python main.py -d data/raw_seq -o out/twopoint -m twopointnuc \
  --opt-twopoint-ratios 1.0,0.5,0.25,0.1 \
  --opt-twopoint-regions 10,20,30
```

## 6. Method Examples

### Two-Point NUC

```bash
python main.py -d data/raw_seq -o out/twopoint -m twopointnuc
```

Disable optimization and use fixed CLI hyperparameters:

```bash
python main.py -d data/raw_seq -o out/twopoint_fixed -m twopointnuc \
  --no-optimize \
  --twopointnuc-num-regions 20 \
  --twopointnuc-lower-percentile 5 \
  --twopointnuc-upper-percentile 95 \
  --twopointnuc-min-valid-ratio 0.8
```

### Multi-Segment NUC

```bash
python main.py -d data/raw_seq -o out/multisegment -m multisegmentnuc
```

### Constant-Statistics NUC

```bash
python main.py -d data/raw_seq -o out/css -m cssbnuc
```

### LMS-SBNUC (gated variant)

```bash
python main.py -d data/raw_seq -o out/lms_gated -m lmssbnuc \
  --lmssbnuc-variant gated
```

## 7. Output

- Corrected frames are saved as:
  - `corrected_0000.png`, `corrected_0001.png`, ...
- Output images are percentile-scaled to 16-bit PNG for display/storage.

## 8. Metrics Used Internally

- `RNU(frame) = std(frame) / mean(frame)` (lower is better)
- `col_mad`, `row_mad`, `spatial_mad` helpers are available in `utils/metrics.py`

## 9. Notes and Limitations

- No formal test suite is included yet.
- Optimization can be slow on long sequences because it runs multiple trial combinations.
- If you only need quick execution, use `--no-optimize` or `--optimize-fast`.

## 10. Troubleshooting

- `No images found`:
  - Check `--data` path and that it contains `.png` files.
- `Expected 16-bit image`:
  - Input frames must be `uint16` PNG.
- Slow runs:
  - Use `--optimize-fast` or reduce candidate lists with `--opt-*` flags.
