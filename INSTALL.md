# Install Fathom

## Before you start

- **Python 3.11+**
- **A device to drive** — an Android device or emulator reachable over `adb` (`adb devices`
  should list it), or an iOS device or simulator behind a WebDriverAgent gateway
- **A vision-language model** — a Gemini API key, or Vertex AI credentials

## 1. Install

```bash
pip install git+https://github.com/DrizzDev/fathom.git
```

Or from a clone, for development:

```bash
git clone https://github.com/DrizzDev/fathom.git
cd fathom
pip install -e .
```

## 2. Configure

Set your model key (and, for Vertex, your project) in the environment or a local `.env`:

```bash
export GEMINI_API_KEY="..."       # or configure Vertex: VERTEX_PROJECT_ID, VERTEX_LOCATION
```

Confirm your device is reachable:

```bash
adb devices        # Android — note the serial for the quickstart
```

## 3. Run

Follow the quickstart in the [README](README.md#quickstart) to wire the device, model, and
perception adapters and run your first intent. For iOS, swap in the `IOSDevice` and
`IOSNativePerceptionAdapter` adapters — they satisfy the same ports.

A `fathom` command-line entrypoint is also installed; run `fathom --help` for its usage.
