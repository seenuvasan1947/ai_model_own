import argparse

TEMPLATE = """Write a spoken narration script about "{topic}" for a YouTube-style voiceover video.

Style rules:
- Conversational, spoken tone (not written/essay style), as if talking directly to the viewer.
- Start with a short greeting/hook line ("Welcome back to the channel...").
- Organize into short paragraphs (2-4 sentences each), separated by a blank line.
- Cover: what it is, why it matters, key facts/stages/steps, and practical takeaways.
- End with a short motivational closing line, then a call to action ("like, share, and subscribe").
- Plain text only, no headings, no bullet points, no markdown.
- Target length: {length} (roughly {seconds}s of narration at normal speaking pace).
"""


def main():
    p = argparse.ArgumentParser(description="Build a narration-script prompt for a given topic.")
    p.add_argument("topic", help="topic to write the script about")
    p.add_argument("--length", default="around 300-400 words", help="rough target length description")
    p.add_argument("--seconds", type=int, default=150, help="rough target narration duration in seconds")
    p.add_argument("--out", help="save the generated prompt to this file instead of only printing it")
    args = p.parse_args()

    prompt = TEMPLATE.format(topic=args.topic, length=args.length, seconds=args.seconds)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Prompt saved to {args.out}\n")

    print(prompt)


if __name__ == "__main__":
    main()
