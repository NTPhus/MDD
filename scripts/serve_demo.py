#!/usr/bin/env python3
"""Serve the MDD demo on localhost, or explicitly allowed private LAN addresses."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mdd.demo import DemoApplication, DemoServer, validate_lan_hosts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--lan", action="store_true", help="Opt in to listening on 0.0.0.0; requires --allow-host")
    parser.add_argument("--allow-host", action="append", default=[], metavar="IPV4",
                        help="Trusted RFC1918 IPv4 address for this machine; repeat for multiple LAN interfaces")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    try:
        allowed_hosts = validate_lan_hosts(args.allow_host, lan=args.lan)
    except ValueError as error:
        parser.error(str(error))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    bind_host = "0.0.0.0" if args.lan else "127.0.0.1"
    server = DemoServer((bind_host, args.port), DemoApplication(ROOT, threads=args.threads),
                        lan=args.lan, allow_hosts=allowed_hosts)
    from mdd.tts_registry import kokoro_status
    if kokoro_status(ROOT)['ready']:
        from scripts.kokoro_tts import prewarm_async
        prewarm_async()
    print(f"Local demo: http://127.0.0.1:{args.port}", flush=True)
    for host in sorted(allowed_hosts):
        print(f"LAN demo: http://{host}:{args.port}", flush=True)
    from mdd.chat_runtime import requested_device
    print(f"Acoustic inference: CPU; Qwen device policy: {requested_device()}. This server does not start or stop training.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
