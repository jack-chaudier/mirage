#!/usr/bin/env python3
"""Prepare clean arXiv upload bundles from canonical RHUN paper sources.

Creates per-paper staging dirs and .tar archives under derived/arxiv_submissions/.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "papers" / "sources" / "rhun"
OUT_ROOT = REPO_ROOT / "derived" / "arxiv_submissions"


@dataclass(frozen=True)
class PaperSpec:
    key: str
    tex_name: str
    bbl_name: str | None


PAPERS: list[PaperSpec] = [
    PaperSpec("paper_01_absorbing_states", "finite_absorbing_states.tex", "finite_absorbing_states.bbl"),
    PaperSpec("paper_02_streaming_oscillation_traps", "streaming_oscillation_traps_draft.tex", None),
    PaperSpec("paper_03_validity_mirage", "context_algebra_draft.tex", "context_algebra_draft.bbl"),
]

STYLE_FILES = ["neurips_2023.sty"]

INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}")
END_DOC_RE = re.compile(r"\\end\{document\}")

TYPEOUT_LINE = r"\typeout{get arXiv to do 4 passes: Label(s) may have changed. Rerun}"


def strip_tex_comments(tex: str) -> str:
    r"""Strip LaTeX comments while preserving escaped percent signs (\%)."""

    out_lines: list[str] = []
    for line in tex.splitlines(keepends=True):
        i = 0
        cut_at = None
        while i < len(line):
            if line[i] == "%":
                backslashes = 0
                j = i - 1
                while j >= 0 and line[j] == "\\":
                    backslashes += 1
                    j -= 1
                if backslashes % 2 == 0:
                    cut_at = i
                    break
            i += 1

        if cut_at is None:
            out_lines.append(line)
        else:
            trimmed = line[:cut_at].rstrip()
            if line.endswith("\n"):
                out_lines.append(trimmed + "\n")
            else:
                out_lines.append(trimmed)

    return "".join(out_lines)


def find_graphics(tex: str) -> list[str]:
    return [m.group(2) for m in INCLUDEGRAPHICS_RE.finditer(tex)]


def flatten_name(src_rel: str, seen: set[str]) -> str:
    base = Path(src_rel).name
    if base not in seen:
        seen.add(base)
        return base

    alt = src_rel.replace("/", "__")
    if alt not in seen:
        seen.add(alt)
        return alt

    stem = Path(base).stem
    suf = Path(base).suffix
    idx = 2
    while True:
        candidate = f"{stem}_{idx}{suf}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        idx += 1


def resolve_source_file(rel_path: str) -> Path:
    src = SRC_DIR / rel_path
    if src.exists():
        return src

    p = Path(rel_path)
    if p.suffix:
        raise FileNotFoundError(f"Missing required figure: {rel_path}")

    for ext in [".pdf", ".png", ".jpg", ".jpeg", ".eps"]:
        cand = SRC_DIR / f"{rel_path}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Missing required figure: {rel_path}")


def rewrite_graphics_paths(tex: str, path_map: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        opts = match.group(1) or ""
        path = match.group(2)
        new_path = path_map.get(path, path)
        return f"\\includegraphics{opts}{{{new_path}}}"

    return INCLUDEGRAPHICS_RE.sub(repl, tex)


def inject_typeout_before_end(tex: str) -> str:
    if TYPEOUT_LINE in tex:
        return tex

    m = END_DOC_RE.search(tex)
    if not m:
        raise ValueError("No \\end{document} found")

    insert = TYPEOUT_LINE + "\n"
    return tex[: m.start()] + insert + tex[m.start() :]


def run_pdflatex(stage_dir: Path, main_tex: str) -> None:
    for _ in range(4):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", main_tex],
            cwd=stage_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


def cleanup_generated(stage_dir: Path, keep_bbl: str | None, main_tex: str) -> None:
    keep = {".tex", ".sty", ".pdf"}  # source pdf figures are needed
    if keep_bbl:
        keep.add(".bbl")

    remove_exts = {
        ".aux",
        ".log",
        ".out",
        ".blg",
        ".toc",
        ".lof",
        ".lot",
        ".fls",
        ".fdb_latexmk",
        ".synctex.gz",
        ".nav",
        ".snm",
    }

    for path in stage_dir.iterdir():
        if path.name.startswith("."):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            continue

        if path.is_dir():
            shutil.rmtree(path)
            continue

        if path.suffix in remove_exts:
            path.unlink()
            continue

        if path.suffix == ".bib":
            path.unlink()
            continue

        if path.suffix == ".bbl" and keep_bbl and path.name == keep_bbl:
            continue

        # Do not include rendered manuscript PDF in arXiv source uploads.
        if path.suffix == ".pdf" and path.stem == Path(main_tex).stem:
            path.unlink()
            continue

        if path.suffix not in keep and path.suffix not in {".bbl"}:
            path.unlink()


def make_tar(stage_dir: Path, tar_path: Path) -> None:
    if tar_path.exists():
        tar_path.unlink()
    subprocess.run(["tar", "-cf", str(tar_path), "."], cwd=stage_dir, check=True)


def prepare_one(spec: PaperSpec) -> tuple[Path, Path]:
    stage_dir = OUT_ROOT / spec.key
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    src_tex = SRC_DIR / spec.tex_name
    if not src_tex.exists():
        raise FileNotFoundError(f"Missing source tex: {src_tex}")

    tex = src_tex.read_text(encoding="utf-8")

    # Copy and flatten graphics.
    needed_graphics = sorted(set(find_graphics(tex)))
    seen_names: set[str] = set()
    path_map: dict[str, str] = {}

    for g in needed_graphics:
        src_file = resolve_source_file(g)
        flat_name = flatten_name(g, seen_names)
        shutil.copy2(src_file, stage_dir / flat_name)
        path_map[g] = flat_name

    tex = rewrite_graphics_paths(tex, path_map)
    tex = strip_tex_comments(tex)
    tex = inject_typeout_before_end(tex)

    # Copy style + bbl + tex.
    for sty in STYLE_FILES:
        sty_src = SRC_DIR / sty
        if not sty_src.exists():
            raise FileNotFoundError(f"Missing style file: {sty_src}")
        shutil.copy2(sty_src, stage_dir / sty)

    if spec.bbl_name:
        bbl_src = SRC_DIR / spec.bbl_name
        if not bbl_src.exists():
            raise FileNotFoundError(f"Missing bbl file: {bbl_src}")
        shutil.copy2(bbl_src, stage_dir / spec.bbl_name)

    (stage_dir / spec.tex_name).write_text(tex, encoding="utf-8")

    # Compile check in stage dir.
    run_pdflatex(stage_dir, spec.tex_name)

    # Remove generated trash; keep only upload-needed files.
    cleanup_generated(stage_dir, spec.bbl_name, spec.tex_name)

    tar_path = OUT_ROOT / f"{spec.key}_arxiv.tar"
    make_tar(stage_dir, tar_path)

    return stage_dir, tar_path


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Source: {SRC_DIR}")
    print(f"Output: {OUT_ROOT}")

    manifest_lines = [
        "# arXiv Upload Bundles",
        "",
        "Generated by `scripts/prepare_arxiv_submissions.py`.",
        "",
    ]

    for spec in PAPERS:
        stage_dir, tar_path = prepare_one(spec)
        files = sorted(p.name for p in stage_dir.iterdir() if p.is_file())
        manifest_lines.extend(
            [
                f"## {spec.key}",
                f"- main_tex: `{spec.tex_name}`",
                f"- tarball: `{tar_path.relative_to(REPO_ROOT)}`",
                f"- staged_dir: `{stage_dir.relative_to(REPO_ROOT)}`",
                "- files:",
            ]
        )
        manifest_lines.extend([f"  - `{f}`" for f in files])
        manifest_lines.append("")
        print(f"Prepared {spec.key}: {tar_path}")

    manifest = OUT_ROOT / "README.md"
    manifest.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
