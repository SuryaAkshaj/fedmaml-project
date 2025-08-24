#!/usr/bin/env python3
"""
Simple runner script for FedMAML project.
Replaces Makefile functionality with Python commands.
"""

import sys
import os
import subprocess

def install():
    """Install dependencies"""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def test():
    """Run tests"""
    print("Running tests...")
    subprocess.run([sys.executable, "run_tests.py"])

def train():
    """Run training"""
    print("Running training...")
    subprocess.run([sys.executable, "train.py"])

def clean():
    """Clean output files"""
    print("Cleaning output files...")
    if os.path.exists("outputs"):
        for file in os.listdir("outputs"):
            os.remove(os.path.join("outputs", file))
    print("Cleaned output files")

def status():
    """Show project status"""
    print("Project Status:")
    py_files = len([f for f in os.listdir(".") if f.endswith(".py")])
    test_files = len([f for f in os.listdir(".") if f.startswith("test_")])
    config_files = len([f for f in os.listdir(".") if f.endswith(".yaml")])
    
    print(f"  Python files: {py_files}")
    print(f"  Test files: {test_files}")
    print(f"  Config files: {config_files}")
    
    if os.path.exists("outputs"):
        output_files = len(os.listdir("outputs"))
        print(f"  Output files: {output_files}")

def help():
    """Show help"""
    print("Available commands:")
    print("  install    - Install dependencies")
    print("  test       - Run all tests")
    print("  train      - Run training with default config")
    print("  clean      - Clean output files")
    print("  status     - Show project status")
    print("  help       - Show this help")

def main():
    if len(sys.argv) < 2:
        help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "install":
        install()
    elif command == "test":
        test()
    elif command == "train":
        train()
    elif command == "clean":
        clean()
    elif command == "status":
        status()
    elif command == "help":
        help()
    else:
        print(f"Unknown command: {command}")
        help()

if __name__ == "__main__":
    main()
