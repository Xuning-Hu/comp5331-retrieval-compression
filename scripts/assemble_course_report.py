"""Assemble the editable report fragments after supplementary evidence is complete."""
from pathlib import Path

from retrieval_lab.common import read_json


def main():
    root = Path(__file__).resolve().parents[1]
    if read_json(root / "docs/verification/supplementary-completion.json")["status"] != "passed":
        raise ValueError("Complete and verify the declared follow-up before report assembly")
    report = root / "docs/report"
    filenames = (
        "00-frontmatter.md", "01-introduction.md", "02-methods.md", "03-protocol.md",
        "04-results.md", "06-rae-verification.md", "06a-rae-followup.md", "06b-supplementary.md",
    )
    parts = [(report / name).read_text(encoding="utf-8").strip() for name in filenames]
    limitations, appendix = (report / "07-scope-and-reproduction.md").read_text(encoding="utf-8").split("## Appendix A.", 1)
    parts += [limitations.strip(), (report / "09-conclusion.md").read_text(encoding="utf-8").strip(),
              (report / "08-references.md").read_text(encoding="utf-8").strip(),
              "## Appendix A." + appendix.rstrip()]
    assembled = "\n\n".join(parts) + "\n"
    (report / "final-report.md").write_text(assembled, encoding="utf-8")
    print(f"Assembled complete report: {len(assembled.split())} whitespace-delimited words")


if __name__ == "__main__":
    main()
