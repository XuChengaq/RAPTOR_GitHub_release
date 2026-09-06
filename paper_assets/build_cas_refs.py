import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "build_reference_doc.py"
OUTPUT = ROOT / "cas-refs.bib"


def load_refs():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "refs" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("refs list not found")


# kind, container, volume, number, pages/article number, year
meta = [
    ("article", "Annual Review of Fluid Mechanics", "52", "", "477--508", "2020"),
    ("article", "AIAA Journal", "58", "1", "25--36", "2020"),
    ("article", "Computational Mechanics", "64", "", "525--545", "2019"),
    ("inproceedings", "Advances in Neural Information Processing Systems", "35", "", "", "2022"),
    ("inproceedings", "Advances in Neural Information Processing Systems", "37", "", "", "2024"),
    ("article", "IEEE Transactions on Neural Networks", "6", "4", "911--917", "1995"),
    ("article", "Nature Machine Intelligence", "3", "", "218--229", "2021"),
    ("inproceedings", "International Conference on Learning Representations", "", "", "", "2021"),
    ("article", "Journal of Machine Learning Research", "24", "89", "1--97", "2023"),
    ("article", "Nature Reviews Physics", "6", "", "320--328", "2024"),
    ("inproceedings", "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition", "", "", "652--660", "2017"),
    ("inproceedings", "Advances in Neural Information Processing Systems", "30", "", "", "2017"),
    ("inproceedings", "Proceedings of the 36th International Conference on Machine Learning", "97", "", "2083--2092", "2019"),
    ("inproceedings", "International Conference on Learning Representations", "", "", "", "2021"),
    ("article", "Journal of Computational Physics", "524", "", "113708", "2025"),
    ("article", "Journal of Machine Learning Research", "24", "388", "1--26", "2023"),
    ("article", "Nature Computational Science", "4", "", "928--940", "2024"),
    ("article", "Computer Methods in Applied Mechanics and Engineering", "443", "", "118022", "2025"),
    ("inproceedings", "Proceedings of the 40th International Conference on Machine Learning", "202", "", "12556--12569", "2023"),
    ("inproceedings", "Proceedings of the 41st International Conference on Machine Learning", "235", "", "53681--53705", "2024"),
    ("inproceedings", "Proceedings of the 42nd International Conference on Machine Learning", "267", "", "41432--41449", "2025"),
    ("article", "Proceedings of the AAAI Conference on Artificial Intelligence", "40", "1", "408--416", "2026"),
    ("misc", "arXiv", "", "", "", "2026"),
    ("article", "Journal of Machine Learning Research", "23", "215", "1--63", "2022"),
    ("article", "Nature Communications", "15", "", "5101", "2024"),
    ("article", "Advances in Water Resources", "163", "", "104180", "2022"),
    ("inproceedings", "Advances in Neural Information Processing Systems", "38", "", "", "2025"),
    ("inproceedings", "Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence", "161", "", "1352--1361", "2021"),
    ("article", "Journal of Computational Design and Engineering", "11", "3", "91--100", "2024"),
    ("article", "Nature Reviews Physics", "3", "", "422--440", "2021"),
    ("article", "Annual Review of Statistics and Its Application", "13", "", "123--148", "2026"),
    ("article", "Journal of Computational Physics", "538", "", "114131", "2025"),
    ("article", "Computer Methods in Applied Mechanics and Engineering", "451", "", "118668", "2026"),
    ("article", "Journal of Computational Physics", "529", "", "113871", "2025"),
    ("article", "ACM/IMS Journal of Data Science", "1", "3", "1--27", "2024"),
]


def bib_escape(text):
    replacements = {
        "&": r"\&",
        "–": "--",
        "—": "---",
        "‑": "-",
        "×": r"$\times$",
        "ö": r'\"{o}',
        "ü": r'\"{u}',
        "é": r"\'{e}",
        "è": r"\`{e}",
        "ören": r'\"{o}ren',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def author_field(authors):
    return " and ".join(bib_escape(a.strip()) for a in authors.split(";"))


def emit_entry(index, ref, info):
    authors, title, _venue, doi, doi_type = ref
    kind, container, volume, number, pages, year = info
    entry_type = "article" if kind == "article" else ("inproceedings" if kind == "inproceedings" else "misc")
    lines = [f"@{entry_type}{{ref{index:02d},", f"  author = {{{author_field(authors)}}},", f"  title = {{{{{bib_escape(title)}}}}},"]
    if kind == "article":
        lines.append(f"  journal = {{{bib_escape(container)}}},")
    elif kind == "inproceedings":
        lines.append(f"  booktitle = {{{bib_escape(container)}}},")
    else:
        lines.extend(["  archivePrefix = {arXiv},", "  eprint = {2608.09764},"])
    if volume:
        lines.append(f"  volume = {{{volume}}},")
    if number:
        lines.append(f"  number = {{{number}}},")
    if pages:
        lines.append(f"  pages = {{{pages}}},")
    lines.append(f"  year = {{{year}}},")
    lines.append(f"  doi = {{{doi}}},")
    lines.append(f"  url = {{https://doi.org/{doi}}},")
    if doi_type == "version":
        lines.append("  note = {The venue does not assign a publisher DOI; the DOI refers to the archived arXiv version},")
    elif doi_type == "preprint":
        lines.append("  note = {Preprint},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


refs = load_refs()
if len(refs) != 35 or len(meta) != 35:
    raise RuntimeError(f"Expected 35 entries, got refs={len(refs)}, meta={len(meta)}")

header = """% RAPTOR reference database
% Citation keys: ref01--ref35
% Complete author lists are retained; no entry uses et al.
% For venues without a publisher DOI, doi/url identify the archived arXiv version.

"""
OUTPUT.write_text(header + "\n\n".join(emit_entry(i, ref, info) for i, (ref, info) in enumerate(zip(refs, meta), 1)) + "\n", encoding="utf-8")
print(OUTPUT)
