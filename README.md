<!--
SPDX-FileCopyrightText: 2025 Patrizia Schoch
SPDX-FileContributor: Hannes Lindenblatt

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Remi Detector Calculation for Monitoring Electrons

GUI tool to simulate Reaction Microscope detector images.

Try out the examples: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HLinde/RedCamel/v0.1.4)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/redcamel)![PyPI - Version](https://img.shields.io/pypi/v/redcamel)![GitHub Release Date](https://img.shields.io/github/release-date/HLinde/RedCamel)![GitHub commits since latest release](https://img.shields.io/github/commits-since/HLinde/RedCamel/latest)
[![REUSE status](https://api.reuse.software/badge/github.com/HLinde/RedCamel)](https://api.reuse.software/info/github.com/HLinde/RedCamel)
![GitHub tag check runs](https://img.shields.io/github/check-runs/HLinde/RedCamel/main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16911941.svg)](https://doi.org/10.5281/zenodo.16911941)

# Example Outputs

![Electron Wiggles](https://codeberg.org/FlashREMI/RedCamel/media/tag/v0.1.4/Electrons.png)
![Ion fragmentation](https://codeberg.org/FlashREMI/RedCamel/media/tag/v0.1.4/Ions.png)

# Usage

## With uv (recommended)

> [!WARNING]
> You currently have to use your system python with included tkinter
> library (e.g. `apt install python3-tk`). uv apparently can not bring a working
> tk library. (Until
> https://github.com/astral-sh/python-build-standalone/pull/676 is merged) If
> you can not get tk to run, you can try to follow the "For developers" section.
> You would notice something like
> `ImportError: module '_tkinter' has no attribute '__file__'`.

Try out with temporary python environment:

```bash
uvx redcamel
```

Permanently install in isolated python environment:

```bash
uv tool install redcamel
```

Run:

```bash
redcamel
```

Upgrade

```bash
uv tool upgrade redcamel
```

uv can be found here: https://docs.astral.sh/uv/getting-started/installation/

## With pipx (recommended)

Install in isolated python environment:

```bash
pipx install redcamel
```

Run:

```bash
redcamel
```

Update:

```bash
pipx upgrade redcamel
```

pipx can be found here: https://pipx.pypa.io/latest/installation/

## With pip

Installing in your current python environment:

Install:

```bash
pip install redcamel
```

Run:

```bash
redcamel
```

Update:

```bash
pip install --upgrade redcamel
```

## With conda/mamba/pixi

Not yet implemented, sorry..

# Authors

- Initial implementation by Patrizia Schoch
- Maintained by Hannes Lindenblatt

# For developers

First get the repository:

```bash
git clone https://codeberg.org/FlashREMI/RedCamel.git
```

or if you have an codeberg.org account with an ssh key set up:

```bash
git clone ssh://git@codeberg.org/FlashREMI/RedCamel.git
```

Then work inside the RedCamel folder. uv or pixi will find their configuration files there.
You might need to get the image files with git-lfs:

```bash
git lfs install
git lfs pull
```

## Usage with pixi (recommended)

> [!TIP]
> pixi / the conda-forge python distribution actually includes tk
> libraries so you do not need installed system tk :)

pixi can be found here: https://pixi.sh/latest/#installation
If you are on a cluster like DESYs maxwell, check if you can load pixi from a module like this:

```bash
module load maxwell pixi
```

Check the available pixi tasks!

```bash
pixi task list
```

Run tasks like this, e.g. the RedCamel:

```bash
pixi run redcamel
```

## Usage with uv

```bash
uv run redcamel
```

uv can be found here: https://docs.astral.sh/uv/getting-started/installation/

## Usage with venv + pip

Set up and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install redcamel in development mode:

```bash
pip install -e ./
```

## Usage with mamba / conda

### Setup

- install environment with dependencies:

```bash
mamba env create
```

### Usage

- activate environment:

```bash
mamba activate redcamel
```

- run GUI with:

```bash
python src/redcamel/remi_gui.py
```

- Play around with plots and sliders!

### Updating

- pull changes:

```bash
git pull
```

- update environment:

```bash
mamba activate redcamel
mamba env update
```
