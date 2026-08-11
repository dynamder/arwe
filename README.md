# arwe

**Arwe** = **Ar**tificial **Wh**ale **E**ye. The whale is a nod to DeepSeek,
the text-only model family this tool is built for.

Experimental CLI that gives text-only LLMs (like DeepSeek) eyes: it applies a
user- or AI-specified convolution kernel to an image and prints the result as
a compact RGB matrix or ASCII art that text-only models can reason about. It
also exposes local OCR (Tesseract) and an optional vision-model layout engine
for frontend debugging.

No machine learning here — just pure 2D convolution over `0..=255` RGB values.

## Build

```bash
cargo build --release
```

## Input

Every subcommand needs exactly one of:

- `--input <path>` — PNG/JPEG/BMP/GIF file
- `--base64 <str>` — base64-encoded image (data-URI prefixes accepted)
- `--stdin` — read base64 from stdin
- `--screen [N]` — capture the primary monitor, or the Nth monitor when `N`
  is given (run with an out-of-range `N` to list available monitors)
- `--window <keyword>` — capture the visible window whose title contains
  `keyword` (case-insensitive substring match; `arwe windows` lists them)

Captures are fed straight into the in-memory pipeline — no temporary PNG
files are written. `--crop`, `--width` and `--sample` behave as usual, with
coordinates relative to the captured frame (monitor or window top-left).

```bash
# primary monitor, local layout analysis
arwe layout --screen

# second monitor (indexes follow (x, y)-sorted monitor order)
arwe layout --screen 1

# window by title keyword, rendered as ASCII at width 80
arwe ascii --window "Chrome" --width 80

# capture straight into the vision layout engine
arwe layout --screen --engine vision
```

`--window` must match exactly one window: on zero or multiple matches arwe
prints the candidates and asks you to narrow the keyword. Minimized windows
and empty titles are skipped. Note: on Linux/Wayland, screen capture is not
supported by the capture backend; the error message will say so.

## Global flags

Available on `matrix`, `ascii`, `convolve` and `layout`:

- `--width N` — area-average downscale to width N (keeps aspect ratio); the
  main token-control lever. Subtle color boundaries (e.g. a light sidebar on
  a white background) survive better at full resolution.
- `--crop X,Y,W,H` — crop in original pixels, applied before `--width`.
- `--sample X,Y` (repeatable) — print `x,y #rrggbb` for pixels in original
  coordinates instead of the normal output. Coordinates are clamped to the
  processed image.
- `--stride N` — take every Nth pixel (matrix/ascii/layout only; `convolve`'s
  `--stride` remains the convolution stride; the vision layout engine ignores
  it).
- `--ansi` — 24-bit ANSI background colors (ascii/layout/convolve ascii output).

Pipeline order: load → crop → width → stride → (optional kernel) → render.
Boxes and samples are always reported in original-image pixel coordinates.

## Subcommands

At a glance:

| Command | What it does |
| --- | --- |
| `matrix` | Compact RGB/grayscale pixel matrix |
| `ascii` | ASCII art (luma through a ramp, optional ANSI) |
| `convolve` | Apply a preset/custom kernel, output matrix or ASCII |
| `layout` | Preset: box-tree layout analysis (vision or convocr engine) |
| `ocr` | Preset: text recognition (vision model or Tesseract) |
| `describe` | Preset: natural-language description (vision model or deterministic digest) |
| `explore` | Free exploration session: probe the image via primitive commands |
| `config` | Show/inspect the vision config file (never prints keys) |
| `presets` | List preset kernels |
| `windows` | List windows capturable with `--window` |

### `matrix`

Print the image as a compact RGB matrix. Add `--grayscale` for single-channel
luma values (Rec. 601: 0.299R + 0.587G + 0.114B).

### `ascii`

Render the image as ASCII art. Default ramp: ` .:-=+*#%@` (dark to bright),
overridable with `--ramp`.

### `convolve`

Apply a kernel and print the result as a matrix or ASCII art.

```bash
# preset kernel
arwe convolve --input demo.png --preset sobel-x --format ascii

# inline JSON, per-channel (applied to R, G, B separately)
arwe convolve --input demo.png \
  --kernel-json '[[-1,0,1],[-2,0,2],[-1,0,1]]' \
  --format matrix

# kernel file (JSON or plain rows of numbers)
arwe convolve --input demo.png --kernel-file kernel.txt --stride 2

# cross-channel kernel: 3D JSON [R][G][B] planes, output is single-channel
arwe convolve --input demo.png \
  --kernel-json '[[[0.299]],[[0.587]],[[0.114]]]'

# grayscale first, then apply a 2D kernel
arwe convolve --input demo.png --preset gaussian3 --grayscale
```

### `layout`

Analyze a UI/page screenshot into a nested box tree via recursive grid
decomposition, for text-only models doing frontend work. A vision-model engine
is available as an alternative (`--engine vision`).

```bash
# JSON + ASCII block map (default)
arwe layout --input shot.png

# JSON only, capped to 32 leaf boxes
arwe layout --input shot.png --format json --max-blocks 32

# ASCII block map with legend only
arwe layout --input shot.png --format ascii

# HTML skeleton with data-x/y/w/h/data-color attributes (deterministic)
arwe layout --input shot.png --format html

# local OCR attached to text nodes (requires Tesseract)
arwe layout --input shot.png --ocr --format json

# vision model layout (defaults to HTML output)
OPENAI_API_KEY=sk-... arwe layout --input shot.png --engine vision

# natural-language layout description (vision engine)
arwe layout --input shot.png --engine vision --format describe

# vision model from a named config profile (see "Vision configuration" below)
arwe layout --input shot.png --engine vision --profile local-vllm

# optional kernel preprocessing (reuses convolve kernels)
arwe layout --input shot.png --preset gaussian3
```

JSON shape (compact, no whitespace):

```json
{"width":1795,"height":1242,"truncated":false,
 "root":{"x":0,"y":0,"w":1795,"h":1242,"kind":"group","direction":"column",
   "color":"#ffffff",
   "children":[
     {"x":0,"y":0,"w":1795,"h":41,"kind":"block","color":"#f9eff9","children":[]},
     {"x":65,"y":41,"w":392,"h":1201,"kind":"block","color":"#ffffff",
      "children":[{"x":90,"y":72,"w":65,"h":19,"kind":"text",
                   "color":"#ffffff","density":0.718,"children":[]}]}
   ]}}
```

Node fields: `x/y/w/h` in original pixels, `kind` (`group|block|text`), `color`
as dominant-color `#rrggbb`, `children[]`. Groups carry `direction`
(`row` = children side by side, `column` = stacked), text nodes carry an edge
`density` and, when `--ocr` was used, the recognized `text`. `truncated:true`
means the `--max-blocks`/`--max-depth` cap was hit.

The ASCII block map prints one legend character per block with a legend
(`A: x,y,w,h #rrggbb [text "…"]`). Text is detected from Laplacian edge energy
with connected-component filtering (compact, dense glyph-like regions only).

#### HTML output (template language)

`--format html` renders the box tree as a nested HTML fragment — the "standard
template language" for frontend debugging. It is deterministic from the local
engine. The vision engine instead emits a complete static HTML document with
CSS and a trailing natural-language description (see `--engine vision` below).

```html
<div data-x="0" data-y="0" data-w="1795" data-h="1242" data-color="#ffffff" data-direction="column">
  <div data-x="0" data-y="0" data-w="1795" data-h="40" data-color="#f0f0f0" data-direction="row">
    <span data-x="10" data-y="10" data-w="120" data-h="20" data-color="#f0f0f0">Fraire</span>
    <div data-x="1695" data-y="0" data-w="100" data-h="40" data-color="#f0f0f0"></div>
  </div>
  <div data-x="0" data-y="40" data-w="310" data-h="1202" data-color="#f5f5f5"></div>
  <div data-x="310" data-y="40" data-w="1485" data-h="1202" data-color="#ffffff"></div>
</div>
```

Rules:

- Only `<div>` and `<span>` are used: groups and blocks become `<div>`, text
  nodes become `<span>`. No classes, ids, styles or other attributes. This
  section describes the deterministic local engine; the vision engine uses a
  richer, document-style HTML format.
- Every element carries `data-x`, `data-y`, `data-w`, `data-h` — integer pixel
  coordinates with origin (0,0) at the top-left of the **original** image —
  and `data-color` with the region's dominant `#rrggbb`.
- Containers whose children are side by side get `data-direction="row"`;
  stacked children get `data-direction="column"`.
- Text nodes carry their OCR text (`arwe layout --ocr`) as the element body,
  HTML-escaped; empty otherwise.
- Nesting reflects containment: children live inside their parent element and
  never overlap.

The vision engine asks the model for boxes in the pixel space of the image it
was sent (`--width` shrinks it) and arwe maps them back to original pixels
exactly, so `json`, `ascii` and `html` output always agree.

#### `--engine vision`

Sends the processed image (controlled by `--width`; `--crop` applies as usual)
to an OpenAI-compatible chat-completions vision model. For `json` output the
coordinates returned by the model are mapped back to original-image pixels
exactly; `html` is a faithful static reproduction sized in original pixels;
`describe` is a natural-language summary.

- `--format html` (the default for vision) returns a complete static HTML
  document that faithfully reproduces the screenshot: semantic class names, a
  `<style>` block with CSS (colors, fonts, spacing, borders, sizes), text
  transcribed character-for-character, and — after the closing `</html>` — a
  natural-language description of the content. It is a static visual
  reproduction: no JavaScript and no functional controls; inputs and buttons
  are represented as styled `<div>`s. Pixel sizes are written in the original
  image's pixel space. Unreadable text is marked `[unreadable]` instead of
  being guessed.
- `--format json` returns the standard layout JSON (vision mode uses the
  model's JSON mode); text nodes carry the recognized `"text"` field.
- `--format describe` returns a natural-language layout description (page
  purpose, section hierarchy, notable content, visual style) as plain text;
  vision engine only.
- `--model`, `--base-url`, `--api-key` override the selected config profile;
  without a profile they fall back to `OPENAI_MODEL`, `OPENAI_BASE_URL`,
  `OPENAI_API_KEY`, then built-in defaults (`gpt-4o-mini`,
  `https://api.openai.com/v1`).
- `--profile <name>` selects a named provider from the config file (falling
  back to `ARWE_PROFILE`, then the config's `default_profile`).
- `--stride` and `--format ascii` are convocr-engine only; `--ocr` requires
  `--engine convocr` (alias `local`).

### Vision configuration

Vision endpoints, models and API keys can be stored in a config file instead
of repeating flags. The file is `ARWE_CONFIG` when set, otherwise
`%APPDATA%\arwe\config.toml` on Windows,
`$XDG_CONFIG_HOME/arwe/config.toml` (or `~/.config/arwe/config.toml`)
on Linux, and `~/Library/Application Support/arwe/config.toml` on macOS.
`arwe config` prints the resolved path and configured profiles;
`arwe config --list` prints profile names only (keys are never shown).

```toml
default_profile = "openai"

[profiles.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key = "sk-..."

# Local vLLM/Ollama servers often need no api_key at all.
[profiles.local-vllm]
base_url = "http://127.0.0.1:8000/v1"
model = "qwen2.5-vl-7b-instruct"
```

Then use it with `--profile local-vllm`; the default profile applies when
`--profile` is omitted. Value precedence:
`--model/--base-url/--api-key` > selected profile > environment variables >
built-in defaults.

### `config`

Inspect the vision config file without revealing secrets:

```bash
arwe config          # resolved path + profile names + default profile
arwe config --path   # resolved config file path only
arwe config --list   # profile names only
```

There is no `--write`/`--init` yet: create the TOML file by hand at the
reported path using the template above.

### `ocr`

Recognize text in the image. Two engines:

- `--engine convocr` (default; alias `local`): local Tesseract. The OCR runs
  on the processed image (`--crop`/`--width` apply), boxes are reported in
  original pixel coordinates.
- `--engine vision`: the vision model transcribes all visible text directly
  (character-for-character; `[unreadable]` marks unreadable text) and returns
  the same JSON/plain shapes — vision lines have no `conf` field.

```bash
# plain text, one line per recognized line
arwe ocr --input shot.png

# JSON with boxes and per-line confidence
arwe ocr --input shot.png --format json

# Chinese + English, custom segmentation mode, cropped region
arwe ocr --input shot.png --lang chi_sim+eng --psm 6 --crop 10,20,400,120

# vision-model OCR (requires a configured provider)
OPENAI_API_KEY=sk-... arwe ocr --input shot.png --engine vision --format json
```

JSON shape:

```json
{"width":1795,"height":1242,
 "lines":[{"x":90,"y":72,"w":65,"h":19,"text":"Login","conf":96.8}]}
```

Tesseract must be installed (`tesseract` on PATH). On Windows, the
UB-Mannheim installer is the common choice. The executable can be overridden
with `ARWE_TESSERACT_BIN` (full path to `tesseract(.exe)`); when it is
missing, the CLI fails with a helpful message instead of a cryptic error.

For layout debugging, `arwe layout --ocr` runs OCR on the original (cropped)
image — before resize/stride — and attaches the recognized text to the matching
text nodes.

### `describe`

Describe the image in natural language. Two engines:

- `--engine vision` (default): the vision model writes a natural-language
  layout description (purpose, sections, content, visual style).
- `--engine convocr` (alias `local`): a deterministic digest assembled from
  the local box tree + OCR text (section counts, hierarchy, recognized text).

```bash
# vision-model description (requires a configured provider)
arwe describe --input shot.png

# offline digest
arwe describe --input shot.png --engine convocr
```

### `explore`

Free exploration session: load the image once and probe it through primitive
commands. Read commands from stdin (interactive when stdin is a TTY; pipe a
batch otherwise); stdout carries only results, the prompt and errors go to
stderr.

```bash
arwe explore --input shot.png --engine vision
```

Primitives:

```text
pixel X,Y                                      sample a pixel color
region [X,Y,W,H] [--width N] [--ascii|--matrix]  view/set the current region
zoom X,Y,W,H                                   set the current region
reset                                          back to the full image
kernel <preset|json> [--stride N] [--ascii|--matrix]
ocr [--lang L] [--psm N]                       Tesseract on the current region
describe [--engine vision|convocr] [--width N]
ask [--engine vision] [--width N] <prompt>     vision model with a custom prompt
help | quit
```

`ask` sends the current region to the vision model with the agent-specified
prompt verbatim (a minimal system message only); it requires the vision engine.
All primitives operate on the current region (initially the full image).

### `presets`

List available preset kernels (`--json` prints matrices as compact JSON, handy
for the AI to pick one).

### `windows`

List the windows `--window` can capture (non-minimized, non-empty titles),
sorted by title:

```bash
arwe windows            # numbered list with titles, apps and geometry
arwe windows --json     # same data as compact JSON
```

Use a title fragment from this list as the `--window` keyword.

## Kernel formats

- 2D array — per-channel kernel: `[[-1,0,1],[-2,0,2],[-1,0,1]]`
- 3D array — cross-channel kernel with exactly 3 input channels (R, G, B):
  `[[[r...]],[[g...]],[[b...]]]`, output is single-channel
- Object with a `"kernel"` field containing either of the above
- Plain-text files: one row per line, numbers separated by whitespace, `,` or
  `|` (2D only)

Kernels are **not normalized** and inputs are **zero-padded**, so blur kernels
like `box3` (sum 9) brighten the image 9x and saturate at 255. That is
intentional: the model controls the exact math.

## Convolution behavior

- Values are `0..=255`; convolution runs on floats.
- Output = `round(sum)`, clamped to `0..=255`.
- Padding is asymmetric `(kh-1)/2` top/left, rest bottom/right, so stride 1
  keeps the input size.
- Output size at stride `s`: `ceil(n / s)`.
- `--stride` supports any value >= 1.

## Output formats

- Matrix (default): no whitespace, `[[[r,g,b],...],...]` for RGB and
  `[[v,...],...]` for single-channel, to keep token counts low.
- ASCII: one character per pixel, luma-mapped through the ramp, no resizing.
- HTML: the local engine emits a nested `<div>`/`<span>` skeleton with
  `data-x/y/w/h/color` (+ `data-direction` on containers); the vision engine
  emits a complete static HTML document with CSS and a trailing
  natural-language description.
- Describe: a natural-language layout description (`arwe layout --engine
  vision --format describe`).

## Environment variables

| Variable | Meaning |
| --- | --- |
| `ARWE_CONFIG` | Path to the vision config file (defaults to the platform config dir) |
| `ARWE_PROFILE` | Default profile name when `--profile` is not passed |
| `OPENAI_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` | Vision fallbacks when no profile matches |
| `ARWE_TESSERACT_BIN` | Full path to `tesseract(.exe)` for OCR |
| `ARWE_DEBUG_VISION` | Set to `1` to print the raw vision-model answer to stderr |

Precedence for vision settings:
`--model/--base-url/--api-key` > selected profile > environment variables >
built-in defaults.

## Troubleshooting & tips

- **Vision JSON output can be unreliable.** Some models ignore the JSON
  instructions and return prose, malformed JSON, or runaway-deep nesting.
  arwe parses with an unbounded recursion limit, rejects layouts nested
  deeper than 128 levels, and salvages the outermost `{...}` object from
  wrapped answers. If a model still fails, prefer `--format html` (the
  default) — set `ARWE_DEBUG_VISION=1` to inspect the raw answer.
- **Empty model replies.** A model occasionally returns no text (more likely
  with small downscaled images). Retry, or drop `--width` and send the image
  at full resolution.
- **Token control.** `--width` is the main lever: a 2500px screenshot sent at
  full resolution costs many tokens. Start with `--width 1200` and raise it
  when text becomes too small to read.
- **Proxies.** `http_proxy`/`https_proxy` are honored. When calling a local
  endpoint (vLLM/Ollama on 127.0.0.1) through a system proxy, set
  `NO_PROXY=127.0.0.1` or the request may fail with 502.
- **Tesseract.** Windows users can install via the UB-Mannheim installer;
  point `ARWE_TESSERACT_BIN` at `tesseract.exe` when it is not on PATH.
- **Config profiles.** A profile may omit `api_key` (local servers), `model`,
  or `base_url`; missing fields fall through to env vars and defaults.

## Try it

```bash
cargo run --example gen_demo      # writes demo.png
cargo run -- matrix --input demo.png
cargo run -- ascii --input demo.png
cargo run -- convolve --input demo.png --preset edge --format ascii
cargo run -- layout --input real_picture/*.png --format ascii
cargo run -- layout --input real_picture/*.png --format html
cargo run -- ocr --input real_picture/*.png --format json
cargo run -- layout --input real_picture/*.png --engine vision
cargo run -- windows
cargo run -- layout --screen --format ascii
```

## Tests

```bash
cargo test
```

## Mutation testing

`cargo-mutants` mutates the source and re-runs the test suite to find
weakly-tested code. Run it with:

```bash
cargo mutants --copy-target true --jobs 8 --exclude 'examples/*'
```

The goal is to keep surviving mutants limited to semantically equivalent
changes (e.g. arithmetic on odd-sized kernel padding that never changes the
result).
