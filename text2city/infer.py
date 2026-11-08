#!/usr/bin/env python3
"""
Week 6: a plain-English sentence -> the 10 parameters -> a generated city.

    python text2city/infer.py "a dense city in the hills with three business hubs" --preview
    python text2city/infer.py "a quiet town on flat land" --dry-run    # print the command only

The model never calls the generator itself. infer.py turns its prediction
into an ordinary generate_city.py command -- the one you've run since
Week 2 -- prints it, and runs it. The printed command is the hand-off
point: copy it, change a number, run it again yourself.
"""
import argparse
import random
import shlex
import subprocess
import sys
from pathlib import Path

# Every path is worked out from this file's own location, so infer.py works
# no matter which folder you run it from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text2city.schema import to_generator_kwargs  # noqa: E402

GENERATE_CITY = ROOT / "scripts" / "generate_city.py"
VISUALIZE_2D = ROOT / "scripts" / "visualize_2d.py"
DEFAULT_CHECKPOINT = ROOT / "text2city" / "checkpoint"
DEFAULT_DATASET = ROOT / "data" / "text2city_dataset.csv"
DEFAULT_OUT = ROOT / "data" / "from_text.npz"

# build_city() keyword arguments whose generate_city.py flag doesn't follow
# the usual rule. Every other flag is "--" + the name with "_" turned into "-".
SPECIAL_FLAGS = {"terrain_archetype": "--terrain", "city_archetype": "--city"}


def predict_params(text, checkpoint_dir=DEFAULT_CHECKPOINT, dataset_path=DEFAULT_DATASET,
                   allow_stale=False) -> dict:
    """One sentence -> {parameter name: predicted value}, floats rounded to 3 decimals."""
    # imported here rather than at the top: torch takes a few seconds to load,
    # and `infer.py --help` shouldn't have to wait for it
    import torch
    from text2city.evaluate import load_checkpoint, predict
    from text2city.freshness import is_stale

    ckpt_file = Path(checkpoint_dir) / "model.pt"
    if not ckpt_file.exists():
        sys.exit(f"No checkpoint at {ckpt_file}. Train one first:  python text2city/train.py")
    fingerprint = torch.load(ckpt_file, map_location="cpu", weights_only=False).get("fingerprint")
    if is_stale(fingerprint, str(dataset_path)) and not allow_stale:
        sys.exit("This checkpoint is STALE: your dataset, text2city code or citygen terrains\n"
                 "changed after it was trained, so its predictions belong to the old setup.\n"
                 "Fix:  python text2city/generate_dataset.py   then   python text2city/train.py\n"
                 "(or pass --allow-stale to use it anyway)")

    # TODO 1: load the trained model and predict this ONE sentence.
    # evaluate.py already has both functions (imported at the top of this function):
    #   load_checkpoint(checkpoint_dir, dataset_path) -> (model, embedder)
    #       pass both paths through str() -- they may arrive as Path objects
    #   predict(model, embedder, list_of_texts) -> a LIST with one dict per text
    model, embedder = load_checkpoint(str(checkpoint_dir), str(dataset_path))  # <-- YOUR CODE HERE
    params = predict(model, embedder, [text])[0]           # <-- YOUR CODE HERE

    return {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()}


def kwargs_to_cli_args(kwargs: dict) -> list:
    """
    build_city() keyword arguments -> generate_city.py command-line arguments.

        {"terrain_archetype": "hills", "clusters": 3, "density": 0.81,
         "water_level": None, "green_override": False}
     -> ["--terrain", "hills", "--clusters", "3", "--density", "0.81"]

    The rules come from how generate_city.py declares its arguments
    (run `python scripts/generate_city.py --help` to see them):
        flag name      SPECIAL_FLAGS if listed there, otherwise
                       "--" + the name with "_" replaced by "-"
        None           leave the flag out -- generate_city.py then uses its
                       own default (water level depends on the terrain)
        True           the flag on its own, no value: --green-override is a
                       switch, it's either there or it isn't
        False          leave it out
        anything else  the flag, then str(value)
    """
    args = []
    for name, value in kwargs.items():
        # TODO 2a: the flag for this keyword argument.
        # Hint: SPECIAL_FLAGS.get(name, fallback) returns fallback when name isn't listed.
        flag = SPECIAL_FLAGS.get(name, "--" + name.replace("_", "-"))  # <-- YOUR CODE HERE

        # TODO 2b: add to `args`, following the four rules in the docstring.
        # Test with `value is None`, `value is True`, `value is False` -- NOT ==.
        # In Python 1 == True and 0.0 == False, so == would turn "--clusters 1"
        # into a bare switch and silently drop "--density 0.0".
        if value is None:
            continue
        elif value is True:
            args.append(flag)
        elif value is False:
            continue
        else:
            args.extend([flag, str(value)])
    return args


def build_command(params: dict, size: int, seed: int, out) -> list:
    """The full generate_city.py command for one set of predicted parameters."""
    # TODO 3: chain the two conversions, then assemble the full command.
    #   params --to_generator_kwargs()--> build_city() keyword arguments
    #          --kwargs_to_cli_args()-->  ["--terrain", "hills", ...]
    # The command is a list of STRINGS, in this order:
    #   sys.executable (the Python running right now), str(GENERATE_CITY),
    #   the args from above, then "--size", "--seed" and "--out", each followed by its value
    kwargs = to_generator_kwargs(params)
    cli_args = kwargs_to_cli_args(kwargs)

    return [
        sys.executable,
        str(GENERATE_CITY),
        *cli_args,
        "--size", str(size),
        "--seed", str(seed),
        "--out", str(out),
    ]


def display_command(cmd: list) -> str:
    """cmd as you'd type it from the repo root: `python scripts/...`, short paths."""
    shown = ["python"]
    for part in cmd[1:]:
        path = Path(part)
        if path.is_absolute() and ROOT in path.parents:
            part = str(path.relative_to(ROOT))
        shown.append(part)
    return shlex.join(shown)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", type=str, help="describe the city, in quotes")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None, help="same seed + same sentence = same city")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--preview", action="store_true", help="also save a 2D PNG next to the .npz")
    parser.add_argument("--dry-run", action="store_true", help="print the command, don't run it")
    parser.add_argument("--allow-stale", action="store_true", help="use a stale checkpoint anyway")
    args = parser.parse_args()

    params = predict_params(args.text, args.checkpoint, args.dataset, args.allow_stale)
    print(f'"{args.text}"\n\npredicted parameters:')
    for name, value in params.items():
        print(f"  {name:20s} = {value}")

    # Always pass a seed -- if you didn't choose one, pick one -- so the
    # printed command rebuilds exactly this city when you run it again.
    seed = args.seed if args.seed is not None else random.randrange(10_000)
    out = Path(args.out).resolve()
    cmd = build_command(params, args.size, seed, out)
    print("\ngenerate_city.py command:\n  " + display_command(cmd))
    if args.dry_run:
        return

    print()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0 or not out.exists():
        sys.exit("generate_city.py failed -- read its error above.")

    if args.preview:
        png = out.with_suffix(".png")
        subprocess.run([sys.executable, str(VISUALIZE_2D), str(out), "--out", str(png)], cwd=ROOT, check=True)
        print(f"preview: {png}")


if __name__ == "__main__":
    main()
