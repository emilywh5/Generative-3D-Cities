# Setup — MDST 3D Generative Pipeline

Follow **Step 1–4** for everyone. **Step 5 is only for Intel Macs** — skip it
otherwise, and do not copy those pinned versions onto a modern machine.

---

## Step 0 — Which path am I on?

On macOS, run:

```bash
uname -m
```

- `arm64` → **Apple Silicon** (M1/M2/M3/M4). Use the standard path. Skip Step 5.
- `x86_64` → **Intel Mac**. You'll likely need Step 5.

Windows and Linux use the standard path.

Also check your Python:

```bash
python3 --version        # 3.10–3.12 recommended
```

---

## Step 1 — Clone and make a virtual environment

```bash
git clone <REPO_URL>
cd MDST-3D-Generator-F26

python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\activate         # Windows PowerShell
```

Your prompt should now start with `(venv)`. **Every command below assumes
the venv is active** — if you open a new terminal, re-run the `activate`
line first.

```bash
pip install --upgrade pip
```

---

## Step 2 — Core dependencies (the city generator)

```bash
pip install -r requirements.txt
```

That's `numpy`, `scipy`, `matplotlib`. Enough to generate and preview cities.

Verify:

```bash
python scripts/generate_city.py --size 128 --terrain hills --city polycentric \
    --clusters 3 --seed 42 --out data/testcity.npz
python scripts/visualize_2d.py data/testcity.npz
```

A 3-panel preview window means Step 2 is done.

---

## Step 3 — ML dependencies (the text-to-parameter model)

Only needed from the ML weeks onward; install now anyway so nobody is
blocked later.

```bash
pip install -r text2city/requirements.txt
```

Verify:

```bash
python -c "import torch, sklearn, sentence_transformers; print('ok', torch.__version__)"
```

If that errors, go to **Step 5**.

---

## Step 4 — Blender

Install Blender from <https://www.blender.org/download/> (4.x is fine; 3.3+
works).

**Typing `blender` in a terminal will almost certainly say
`command not found`** — Blender doesn't add itself to your PATH. That's
normal. Two options:

**Option A — use the full path** (no setup, just verbose):

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --version
```

**Option B — add a shortcut** (recommended):

```bash
# macOS, zsh (default shell)
echo 'alias blender="/Applications/Blender.app/Contents/MacOS/Blender"' >> ~/.zshrc
source ~/.zshrc
blender --version
```

Use `~/.bash_profile` instead if `echo $SHELL` says bash.

On Windows the executable is at something like
`C:\Program Files\Blender Foundation\Blender 4.2\blender.exe`.

> The Week 3 notebook auto-detects Blender in the usual install locations,
> so Option B isn't strictly required for the notebook — but it makes
> running things from a terminal much less painful.

---

## Step 5 — Intel Mac only: version pins

**Skip this entirely on Apple Silicon, Windows, or Linux.** These pins
*downgrade* packages and exist purely to work around Intel Mac limits.

### Why

PyTorch [deprecated macOS x86 builds starting January 2024](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690),
and **2.2.2 is the last version with an Intel Mac wheel**. On an Intel Mac,
`pip install torch` silently caps at 2.2.2 and `pip install "torch>=2.5"`
fails with "Could not find a version that satisfies the requirement". That
one constraint causes two knock-on problems:

1. **NumPy ABI mismatch.** torch 2.2.2 was compiled against NumPy 1.x. With
   NumPy 2.x installed you get `_ARRAY_API not found` warnings and then
   `RuntimeError: Could not infer dtype of numpy.float32`.
2. **transformers refuses to use PyTorch.** Recent `transformers` releases
   check the torch version and disable their PyTorch backend below a rising
   minimum (2.4, then 2.5, then higher) — unsatisfiable at 2.2.2. Symptom:
   `[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but
   found 2.2.2`, sometimes followed by a hard crash
   (`NameError: name 'nn' is not defined` inside
   `transformers/integrations/accelerate.py`).

### The fix

```bash
pip install "numpy<2"

pip uninstall -y sentence-transformers transformers huggingface-hub tokenizers safetensors
pip install "sentence-transformers==2.2.2" "transformers==4.30.2" "huggingface-hub==0.16.4"
```

Verify:

```bash
python -c "import torch, numpy, sentence_transformers; print(torch.__version__, numpy.__version__)"
```

Expect roughly `2.2.2 1.26.x` with no warnings.

### Known-good Intel Mac combination

| package | version |
|---|---|
| Python | 3.11 |
| torch | 2.2.2 |
| numpy | < 2 |
| sentence-transformers | 2.2.2 |
| transformers | 4.30.2 |
| huggingface-hub | 0.16.4 |
| Blender | 3.3+ |

---

## Step 6 — First model run (one-time download)

The first time you train with the sentence-transformer embedder, it
downloads the `all-MiniLM-L6-v2` weights (~80 MB). **You need internet for
that one run**; it's cached locally afterwards and runs fully offline.

```bash
python text2city/generate_dataset.py --n 15000 --seed 0 --out data/text2city_dataset.csv
python text2city/train.py --embedder sentence-transformer
```

No internet, or Step 5 won't cooperate? There's a fully offline fallback
that needs no downloads:

```bash
python text2city/train.py --embedder tfidf
```

---

## Troubleshooting

**`command not found: blender`** → Step 4. Expected, not broken.

**`ModuleNotFoundError: No module named 'torch'`** → you installed
`requirements.txt` but not `text2city/requirements.txt`. They're separate
files (Step 3). Note that `pip install -r` treats an already-installed
package as satisfied and won't upgrade it.

**`Could not infer dtype of numpy.float32`** → Step 5.

**`[transformers] Disabling PyTorch...`** → Step 5.

**`Could not find a version that satisfies the requirement torch>=2.5`** →
you're on an Intel Mac. 2.2.2 is the ceiling; don't fight it, use Step 5.

**`Error: The "filepath" property was not an absolute path`** → a Blender
script passing a relative save path. Needs `Path(...).resolve()`.

**`ModuleNotFoundError: No module named 'scipy'` *inside Blender*** →
Blender bundles numpy but **not** scipy, and installing it into Blender's
bundled Python is painful. Anything that runs inside Blender must use numpy
and the standard library only.

**Blender opens the `.blend` but everything is grey** → the viewport's
default **Solid** shading ignores materials. Switch the shading dropdown
(top-right of the 3D viewport) to **Material Preview**.

**Everything broke, start over:**

```bash
deactivate
rm -rf venv
# then from Step 1
```
