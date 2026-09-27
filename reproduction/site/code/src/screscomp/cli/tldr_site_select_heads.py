from __future__ import annotations

import argparse
import json
from pathlib import Path

from screscomp.modeling import TransformersABBackend
from screscomp.site.model import PreOInterface
from screscomp.tldr_site.protocol import SELECTORS, TldrSiteProtocol, require
from screscomp.tldr_site.selection import run_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one locked TLDR Site selector.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selector", choices=SELECTORS, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = TldrSiteProtocol.load(args.config, validate_files=False)
    output = protocol.paths.selector_dir(args.selector)
    completed = protocol.paths.candidates_json(args.selector)
    if completed.is_file():
        payload = json.loads(completed.read_text(encoding="utf-8"))
        require(payload.get("config_sha256") == protocol.config_sha256, "existing selector config hash drift")
        require(payload.get("status") == "complete", "existing selector artifact is incomplete")
        print(f"[tldr-site-selector] reuse selector={args.selector} output={completed}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.selector == "random":
        run_selector(protocol, args.selector, None)
    else:
        backend = TransformersABBackend(
            model_name_or_path=str(protocol.model_path),
            tokenizer_name_or_path=str(protocol.data["model"]["tokenizer_path"]),
            device=args.device,
            use_chat_template=False,
            torch_dtype=protocol.data["model"]["torch_dtype"],
        )
        run_selector(protocol, args.selector, PreOInterface(backend))
    require(completed.is_file(), f"selector did not emit {completed}")
    print(f"[tldr-site-selector] complete selector={args.selector} output={completed}", flush=True)


if __name__ == "__main__":
    main()
