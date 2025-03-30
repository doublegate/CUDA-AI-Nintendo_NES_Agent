#!/usr/bin/env python3
"""
CUDA-AI-Nintendo_NES_Agent - Agent Runner Script

This script serves as a convenient wrapper to run the CUDA-accelerated
AI agent for NES emulation with various command-line options.

It handles configuration loading, environment setup, and agent execution.

Author: doublegate
License: MIT
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the CUDA-AI Nintendo NES Agent")
    parser.add_argument("--config", type=str, default="../config/agent_config.json",
                        help="Path to the agent configuration file")
    parser.add_argument("--rom", type=str, help="Path to the NES ROM file (overrides config)")
    parser.add_argument("--mode", type=str, choices=["train", "test", "visualize"], default="train",
                        help="Agent operation mode")
    parser.add_argument("--checkpoint", type=str, help="Path to the agent checkpoint to load")
    parser.add_argument("--episodes", type=int, help="Number of episodes to run (overrides config)")
    parser.add_argument("--device", type=int, help="CUDA device ID (overrides config)")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--record", action="store_true", help="Record agent gameplay")
    parser.add_argument("--record-path", type=str, default="../recordings",
                        help="Path to save recordings")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()

def load_config(config_path):
    """Load the agent configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def find_executable():
    """Find the agent executable."""
    # Try common binary locations
    script_dir = Path(__file__).resolve().parent
    
    # Debug build
    debug_path = script_dir.parent / "target" / "debug" / "cuda_ai_nintendo_nes_agent"
    if sys.platform == "win32":
        debug_path = debug_path.with_suffix(".exe")
        
    # Release build
    release_path = script_dir.parent / "target" / "release" / "cuda_ai_nintendo_nes_agent"
    if sys.platform == "win32":
        release_path = release_path.with_suffix(".exe")
    
    if release_path.exists():
        return str(release_path)
    elif debug_path.exists():
        return str(debug_path)
    else:
        print("Agent executable not found. Make sure to build the project first.")
        print(f"Expected at: {release_path} or {debug_path}")
        sys.exit(1)

def prepare_environment(config, args):
    """Prepare the environment for the agent."""
    # Ensure required directories exist
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    os.makedirs(project_dir / config["emulation"]["save_state_path"], exist_ok=True)
    os.makedirs(project_dir / config["logging"]["log_path"], exist_ok=True)
    
    if config["logging"]["tensorboard"]:
        os.makedirs(project_dir / config["logging"]["tensorboard_path"], exist_ok=True)
    
    if args.record:
        os.makedirs(args.record_path, exist_ok=True)
    
    # Set up environment variables
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["cuda"]["device_id"])
    
    # Return the modified configuration
    modified_config = config.copy()
    
    if args.rom:
        modified_config["emulation"]["rom_path"] = args.rom
    
    if args.episodes:
        modified_config["training"]["episodes"] = args.episodes
        
    return modified_config

def build_command(executable, config, args):
    """Build the command to run the agent."""
    cmd = [executable]
    
    # Basic arguments
    cmd.extend(["--mode", args.mode])
    cmd.extend(["--config", args.config])
    
    # ROM path
    if args.rom:
        cmd.extend(["--rom", args.rom])
    else:
        cmd.extend(["--rom", config["emulation"]["rom_path"]])
    
    # Checkpoint
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    
    # Episodes
    if args.episodes:
        cmd.extend(["--episodes", str(args.episodes)])
    
    # Rendering
    if args.render or args.mode in ["test", "visualize"]:
        cmd.append("--render")
    
    # Recording
    if args.record:
        cmd.append("--record")
        cmd.extend(["--record-path", args.record_path])
    
    # Verbosity
    if args.verbose:
        cmd.append("--verbose")
    
    return cmd

def run_agent(cmd):
    """Run the agent with the specified command."""
    print(f"Running agent with command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
        
        return process.poll()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating agent...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Agent did not terminate gracefully, killing process...")
            process.kill()
        return 1

def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)
    config = prepare_environment(config, args)
    executable = find_executable()
    cmd = build_command(executable, config, args)
    
    start_time = time.time()
    exit_code = run_agent(cmd)
    end_time = time.time()
    
    print(f"\nAgent execution completed in {end_time - start_time:.2f} seconds with exit code {exit_code}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
