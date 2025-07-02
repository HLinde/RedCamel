<!--
SPDX-FileCopyrightText: 2025 Patrizia Schoch
SPDX-FileContributor: Hannes Lindenblatt

SPDX-License-Identifier: GPL-3.0-or-later
-->
# Remi Detector Calculation Toolkit $${\color{white}ten}$$

GUI tool to simulate Reaction Microscope detector images.
# Usage with pixi
```bash
pixi run red-cat
```
pixi can be found here: https://pixi.sh/latest/#installation

# Usage with uv
```bash
uv run red-cat
```
uv can be found here: https://docs.astral.sh/uv/getting-started/installation/

# Example Outputs
![Electron Wiggles](Electrons.png)
![Ion fragmentation](Ions.png)

# Usage with mamba / conda
## Setup
- install environment with dependencies:
```bash
mamba env create
```
## Usage
- activate environment:
```bash
mamba activate red-cat
```
- run GUI with:
```bash
python src/red_cat/remi_gui.py
```
- Play around with plots and sliders!

## Updating
- pull changes:
```bash
git pull
```
- update environment:
```bash
mamba activate red-cat
mamba env update
```

# Authors
- Initial implementation by Patrizia Schoch
- Maintained by Hannes Lindenblatt
