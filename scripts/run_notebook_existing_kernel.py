#!/usr/bin/env python3
"""Execute a notebook in an already-running Jupyter kernel and save outputs.

Unlike nbconvert/nbclient's usual execution path, this script never creates or
shuts down a kernel. It is intended for large in-memory models that must be
reused across notebooks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from jupyter_client import BlockingKernelClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--connection-file", type=Path, required=True)
    parser.add_argument("--start-cell", type=int, default=0)
    parser.add_argument("--stop-cell", type=int)
    parser.add_argument("--ready-timeout", type=float, default=60)
    parser.add_argument("--message-timeout", type=float, default=120)
    parser.add_argument("--save-interval", type=float, default=20)
    return parser.parse_args()


def atomic_save(notebook: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".executing.tmp")
    temporary.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_stream(outputs: list[dict[str, Any]], name: str, text: str) -> None:
    if outputs and outputs[-1].get("output_type") == "stream" and outputs[-1].get("name") == name:
        outputs[-1]["text"] += text
    else:
        outputs.append({"output_type": "stream", "name": name, "text": text})


def output_from_message(message: dict[str, Any]) -> dict[str, Any] | None:
    message_type = message["header"]["msg_type"]
    content = message["content"]
    if message_type == "execute_result":
        return {
            "output_type": "execute_result",
            "execution_count": content.get("execution_count"),
            "data": content.get("data", {}),
            "metadata": content.get("metadata", {}),
        }
    if message_type == "display_data":
        return {
            "output_type": "display_data",
            "data": content.get("data", {}),
            "metadata": content.get("metadata", {}),
        }
    if message_type == "error":
        return {
            "output_type": "error",
            "ename": content.get("ename", "Error"),
            "evalue": content.get("evalue", ""),
            "traceback": content.get("traceback", []),
        }
    return None


def matching_shell_reply(
    client: BlockingKernelClient,
    message_id: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    while True:
        reply = client.get_shell_msg(timeout=timeout)
        if reply.get("parent_header", {}).get("msg_id") == message_id:
            return reply


def execute_cell(
    client: BlockingKernelClient,
    notebook: dict[str, Any],
    notebook_path: Path,
    cell_index: int,
    *,
    message_timeout: float,
    save_interval: float,
) -> None:
    cell = notebook["cells"][cell_index]
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    cell["outputs"] = []
    cell["execution_count"] = None
    outputs = cell["outputs"]
    message_id = client.execute(source, store_history=True, stop_on_error=True)
    last_save = time.monotonic()
    saw_idle = False
    saw_error = False

    while not saw_idle:
        message = client.get_iopub_msg(timeout=message_timeout)
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        message_type = message["header"]["msg_type"]
        content = message["content"]
        if message_type == "status" and content.get("execution_state") == "idle":
            saw_idle = True
            continue
        if message_type == "stream":
            stream_text = content.get("text", "")
            append_stream(outputs, content.get("name", "stdout"), stream_text)
            sys.stdout.write(stream_text)
            sys.stdout.flush()
        elif message_type == "clear_output":
            outputs.clear()
        elif message_type in {"execute_result", "display_data", "error"}:
            output = output_from_message(message)
            if output is not None:
                outputs.append(output)
            if message_type == "error":
                saw_error = True
                traceback = "\n".join(content.get("traceback", []))
                print(traceback, file=sys.stderr, flush=True)
        if time.monotonic() - last_save >= save_interval:
            atomic_save(notebook, notebook_path)
            last_save = time.monotonic()

    reply = matching_shell_reply(client, message_id, timeout=message_timeout)
    cell["execution_count"] = reply.get("content", {}).get("execution_count")
    atomic_save(notebook, notebook_path)
    if saw_error or reply.get("content", {}).get("status") != "ok":
        raise RuntimeError(
            f"Cell {cell_index} failed: {reply.get('content', {}).get('ename', 'error')} "
            f"{reply.get('content', {}).get('evalue', '')}"
        )


def main() -> int:
    args = parse_args()
    notebook_path = args.notebook.resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    client = BlockingKernelClient()
    client.load_connection_file(str(args.connection_file))
    client.start_channels()
    try:
        client.wait_for_ready(timeout=args.ready_timeout)
        stop = args.stop_cell if args.stop_cell is not None else len(notebook["cells"])
        for cell_index in range(args.start_cell, min(stop, len(notebook["cells"]))):
            cell = notebook["cells"][cell_index]
            if cell.get("cell_type") != "code":
                continue
            print(f"\n=== executing cell {cell_index}/{len(notebook['cells']) - 1} ===", flush=True)
            execute_cell(
                client,
                notebook,
                notebook_path,
                cell_index,
                message_timeout=args.message_timeout,
                save_interval=args.save_interval,
            )
        print("\nNotebook execution complete:", notebook_path, flush=True)
        return 0
    finally:
        client.stop_channels()


if __name__ == "__main__":
    raise SystemExit(main())
