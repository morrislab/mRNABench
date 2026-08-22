# mRNABench logo assets

SVG, transparent background — swap page backgrounds freely.

## Colours
- ink black `#0B0B0C` + RNA red `#E5232B` → files ending `-black`
- white `#FFFFFF` + RNA red `#FF4A4A` → files ending `-white`

## What's here

| file | use |
|---|---|
| `mark/mrnabench-5bar-*` (scalable) | primary mark, any size |
| `mark/64px · 40px · 24px/mrnabench-5bar-*` | UI at fixed sizes |
| `mark/16px/mrnabench-5bar-16-*` | four-bar reduction for 16px |
| `mark/mrnabench-7bar-*` (scalable) | print / large display mark |
| `mark/64px · 40px/mrnabench-7bar-*` | large only — below 40px use 5bar |
| `mark/mrnabench-hex5-*` (scalable) | hexagon badge — the 5-bar mark at 72% inside the shell |
| `mark/64px · 40px/mrnabench-hex5-*` | hexagon badge, fixed sizes (40px floor) |
| `mark/24px/mrnabench-hex-24-*` | hexagon badge reduced to 3 reads |
| `mark/16px/mrnabench-hex-16-*` | hexagon badge reduced to 1 read |
| `lockup/mrnabench-lockup-5bar-*` | horizontal logo, UI |
| `lockup/mrnabench-lockup-7bar-*` | horizontal logo, print / hero |
| `lockup/*-outlined.svg` | fixed Archivo outlines for README and image use |
| `favicon/` | 32px bare + hexagon |

## Reduction rules
- 7 bar → 5 bar below 40px
- 5 bar → 4 bar below 24px
- hexagon badge: 5 reads at 40px and up, 3 reads at 24px, 1 read at 16px

Use the bare mark by default; the hexagon versions are for badges, app icons and
avatars where the logo needs a hard edge against a busy background.

## Type
Wordmark: **Archivo 500, −3% tracking**. m + RNA (red) + Bench, no space.
Lockup SVGs use live text — load Archivo (Google Fonts) or it falls back to
Helvetica/Arial. On the web, prefer inline SVG for the mark and real HTML text
for the wordmark so it stays selectable and restyles for dark mode.

Use the `-outlined.svg` lockups where the font cannot be loaded, including
GitHub README images. These preserve the Archivo letterforms as vector paths.
