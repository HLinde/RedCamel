<!--
SPDX-FileCopyrightText: 2025 Patrizia Schoch
SPDX-FileContributor: Hannes Lindenblatt

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Remi Detector Calculation for Monitoring Electrons

GUI tool to simulate Reaction Microscope detector images.

# Example Outputs

![Electron Wiggles](https://codeberg.org/FlashREMI/RedCamel/media/branch/main/Electrons.png)
![Ion fragmentation](https://codeberg.org/FlashREMI/RedCamel/media/branch/main/Ions.png)

# Usage

## With uv (recommended)

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

## With conda/mamba/pipx

Not yet implemented, sorry..

# Authors

- Initial implementation by Patrizia Schoch
- Maintained by Hannes Lindenblatt

# For developers

## Usage with pixi

```bash
pixi run redcamel
```

pixi can be found here: https://pixi.sh/latest/#installation

## Usage with uv

```bash
uv run redcamel
```

uv can be found here: https://docs.astral.sh/uv/getting-started/installation/

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
