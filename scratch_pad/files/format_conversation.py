#!/usr/bin/env python3
"""
Formatter for Claude conversation JSON exports.
Strips metadata and keeps only the conversation text.

Usage:
    python format_conversation.py                    # List files, choose one
    python format_conversation.py message_2.txt      # Format specific file
    python format_conversation.py message_2.txt -o cleaned.txt  # Save to custom file
"""

import json
import sys
import os
from pathlib import Path


def list_files(directory="."):
    """List .txt and .json files in directory."""
    files = [f for f in os.listdir(directory)
             if f.endswith((".txt", ".json")) and f != "format_conversation.py"]
    return sorted(files)


def choose_file(files):
    """Display file list and let user choose."""
    if not files:
        print("No .txt or .json files found in current directory.")
        return None

    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    try:
        choice = int(input("\nChoose file (number): "))
        if 1 <= choice <= len(files):
            return files[choice - 1]
    except (ValueError, IndexError):
        pass

    print("Invalid choice.")
    return None


def parse_conversation_json(content):
    """Extract conversation messages from JSON export."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    messages = []
    chat_messages = data.get("tree", {}).get("chat_messages", [])

    for msg in chat_messages:
        sender = msg.get("sender", "unknown")
        content_blocks = msg.get("content", [])

        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        text = "".join(text_parts).strip()
        if text:
            messages.append((sender, text))

    return messages


def format_to_text(messages):
    """Format conversation as readable text."""
    output = []
    for sender, text in messages:
        prefix = "USER:" if sender == "human" else "CLAUDE:"
        output.append(f"\n{prefix}\n{text}\n")
    return "".join(output)


def main():
    input_file = None
    output_file = None

    # Parse arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if len(sys.argv) > 3 and sys.argv[2] == "-o":
            output_file = sys.argv[3]

    # Choose file if not provided
    if not input_file:
        files = list_files()
        input_file = choose_file(files)
        if not input_file:
            return

    # Read file
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse conversation
    messages = parse_conversation_json(content)
    if not messages:
        print("Could not parse conversation from file.")
        return

    # Format output
    formatted = format_to_text(messages)

    # Save or print
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"\n[OK] Formatted conversation saved to: {output_file}")
    else:
        # Default: save as [input]_clean.txt
        base = Path(input_file).stem
        output_file = f"{base}_clean.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"\n[OK] Formatted conversation saved to: {output_file}")

    print(f"  {len(messages)} messages extracted")


if __name__ == "__main__":
    main()
