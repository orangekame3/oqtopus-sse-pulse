# oqtopus-sse-pulse

## Prequisite

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Local Setup

```bash
git clone https://github.com/orangekame3/oqtopus-sse-pulse.git
cd oqtopus-sse-pulse
uv sync
```

## GitHub Codespaces Setup

1. Fork this repository
2. Setup codespaces secret^[1]
3. Open repository in GitHub Codespaces
4. Install uv with `pip install uv`
5. `uv sync`
* [1] codespaces secret
```bash
OQTOPUS_URL=https://api.qiqb-cloud.jp
OQTOPUS_API_TOKEN=xxxxxxxxxxxx
```


## Usage

```bash
mkdir workspace
cp -rf examples/* workspace/
```

## Note

To restrict access qubit, provider have to remove qubit frequency from props.yaml

## Your Config Data

https://github.com/orangekame3/oqtopus-pulse-config
