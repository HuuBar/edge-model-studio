#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Start vLLM server via YAML config.

YAML fields must match vLLM CLI arguments (without --).
Script automatically prepends -- when building commands.

Example:
    python run_online_server.py --config benchmark.yaml
    python run_online_server.py --config benchmark.yaml --dry-run

    # To stop server:
    pkill -f "vllm serve"
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


def parse_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Parse and return YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise SystemExit(f"Config file must be a YAML object, got {type(config).__name__}")
    return config


def validate_server_config(config: Dict[str, Any]) -> List[str]:
    """Validate config and return list of error messages."""
    errors = []

    if "server" not in config:
        errors.append("Missing required section: server")
    else:
        server = config["server"]
        if not isinstance(server, dict):
            errors.append("server must be a dictionary")
        elif "model" not in server:
            errors.append("server.model is required")

    return errors


def set_environment_vars(config: Dict[str, Any]) -> None:
    """Set environment variables from config before running server."""
    if "env" not in config:
        return

    env_vars = config["env"]
    if not isinstance(env_vars, dict):
        print(f"[WARNING] env section must be a dictionary, skipping environment variables")
        return

    print("\n" + "=" * 60)
    print("Environment Variables")
    print("=" * 60)
    for key, value in env_vars.items():
        os.environ[key] = str(value)
        print(f"  {key}={value}")
    print("=" * 60 + "\n")


def build_server_command(server: Dict[str, Any]) -> List[str]:
    """Build vllm serve command from server config.
    
    YAML fields directly match vLLM CLI arguments (with --).
    """
    cmd = ["vllm", "serve", server["model"]]

    for key, value in server.items():
        if key == "model":
            continue
        if value is None:
            continue

        cli_arg = f"--{key}"

        if isinstance(value, bool):
            if value:
                cmd.append(cli_arg)
        elif isinstance(value, (dict, list)):
            # Nested objects like compilation_config -> JSON string
            cmd.append(cli_arg)
            cmd.append(json.dumps(value))
        else:
            cmd.append(cli_arg)
            cmd.append(str(value))

    return cmd


def print_command(cmd: List[str]) -> None:
    """Print a command in a reproducible format."""
    print("\n" + "=" * 60)
    print("Command:")
    print("=" * 60)
    print(" ".join(shlex.quote(arg) for arg in cmd))
    print("=" * 60 + "\n")


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for server to be ready by checking if the port is open."""
    print(f"[INFO] Waiting for server on port {port} to be ready...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                time.sleep(2)
                print(f"[INFO] Server is ready!")
                return True
        except Exception:
            pass
        time.sleep(5)

    print(f"[ERROR] Server did not become ready within {timeout}s")
    return False


def print_config_summary(server: Dict[str, Any]) -> None:
    """Print a summary of the server config."""
    print("\n" + "=" * 60)
    print("Server Config Summary")
    print("=" * 60)
    for key, value in server.items():
        if isinstance(value, (dict, list)):
            print(f"  {key}: {json.dumps(value)}")
        else:
            print(f"  {key}: {value}")
    print("\n" + "=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Start vLLM server via YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with config
  python run_online_server.py --config benchmark.yaml

  # Dry run (print command without executing)
  python run_online_server.py --config benchmark.yaml --dry-run
        """,
    )
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--dry-run", action="store_true", help="Print command without executing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse config
    try:
        config = parse_yaml_config(args.config)
    except FileNotFoundError:
        raise SystemExit(f"Config file not found: {args.config}")
    except yaml.YAMLError as exc:
        raise SystemExit(f"Failed to parse YAML: {exc}")

    # Validate config
    errors = validate_server_config(config)
    if errors:
        print("Config validation failed:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    # Set environment variables
    set_environment_vars(config)

    server = config["server"]
    print_config_summary(server)

    cmd = build_server_command(server)

    if args.dry_run:
        print("[DRY RUN] Server command:")
        print_command(cmd)
        return

    # Start server - logs go directly to console (no capture)
    print("[INFO] Starting server...")
    print_command(cmd)

    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,  # Inherit parent stdout to show logs on console
        stderr=subprocess.STDOUT,
    )

    port = server.get("port", 8000)
    if not wait_for_server(port):
        print("[ERROR] Server failed to start")
        process.terminate()
        raise SystemExit(1)

    print("[INFO] Server is running. Logs are displayed above.")
    print("[INFO] To stop server:")
    print(f"  pkill -f 'vllm serve'")
    print(f"  # or")
    print(f"  kill {process.pid}")

    # Wait for server process to finish (or interrupt)
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Received interrupt, stopping server...")
        process.terminate()
        process.wait()
        print("[INFO] Server stopped")


if __name__ == "__main__":
    main()
