#!/usr/bin/env python3
"""커밋 메시지 규칙 검사 (GUIDE.md §4, Conventional Commits).

사용법:
    python3 scripts/check_commit_msg.py <메시지 파일>     # commit-msg 훅
    python3 scripts/check_commit_msg.py --range A..B     # CI: 범위 안의 커밋 전부
"""

import re
import subprocess
import sys

TYPES = ("feat", "fix", "refactor", "docs", "chore", "ci")
SCOPE_REQUIRED_TYPES = ("feat", "fix", "refactor")
SCOPE_PATTERN = r"Day\d{2}|\d{4}-\d{2}-\d{2}|Project\d{2}"
SUBJECT_PATTERN = re.compile(
    rf"^(?P<type>{'|'.join(TYPES)})"
    rf"(\((?P<scope>{SCOPE_PATTERN})\))?"
    r": (?P<summary>.+)$"
)
MAX_SUBJECT_LENGTH = 72
VAGUE_SUMMARIES = {
    "update", "fix", "test", "wip", "edit", "change", "changes", "commit",
    "fix bug", "fix bugs", "update code", "update files", "minor changes",
}
HANGUL_PATTERN = re.compile(r"[가-힣ㄱ-ㆎ]")
FORBIDDEN_TRAILER_PATTERN = re.compile(r"^co-authored-by:", re.IGNORECASE | re.MULTILINE)
SKIP_PREFIXES = ("Merge ", "Revert ")


def check_message(message):
    """커밋 메시지 하나를 검사해 위반 사항 목록을 돌려준다 (비어 있으면 통과)."""
    lines = [line for line in message.splitlines() if not line.startswith("#")]
    subject = lines[0].strip() if lines else ""

    if subject.startswith(SKIP_PREFIXES):
        return []

    errors = []
    match = SUBJECT_PATTERN.match(subject)
    if not match:
        errors.append(
            "형식은 '<type>(<scope>): <summary>' 이어야 함 "
            f"(type: {', '.join(TYPES)} / scope: DayNN, YYYY-MM-DD, ProjectNN)"
        )
    else:
        commit_type = match.group("type")
        summary = match.group("summary")
        if commit_type in SCOPE_REQUIRED_TYPES and not match.group("scope"):
            errors.append(f"'{commit_type}' 커밋에는 scope가 필요함 (예: {commit_type}(Day01): ...)")
        if summary.lower().strip(" .") in VAGUE_SUMMARIES or len(summary.split()) < 2:
            errors.append(f"요약 '{summary}'이(가) 모호함 — 무엇을 했는지 구체적으로 작성")
        if summary.endswith("."):
            errors.append("요약 끝에 마침표를 붙이지 않음")

    if len(subject) > MAX_SUBJECT_LENGTH:
        errors.append(f"첫 줄이 {len(subject)}자 — {MAX_SUBJECT_LENGTH}자 이하로 줄일 것")
    if HANGUL_PATTERN.search(subject):
        errors.append("첫 줄은 영어로 작성")
    if FORBIDDEN_TRAILER_PATTERN.search("\n".join(lines)):
        errors.append("Co-Authored-By 트레일러는 넣지 않음")
    return errors


def commits_in_range(revision_range):
    """범위 안의 (커밋 해시, 메시지) 목록을 돌려준다."""
    output = subprocess.run(
        ["git", "log", "--format=%H%x00%B%x1e", revision_range],
        check=True, capture_output=True, text=True,
    ).stdout
    commits = []
    for record in output.split("\x1e"):
        if record.strip():
            commit_hash, message = record.strip().split("\x00", 1)
            commits.append((commit_hash[:7], message))
    return commits


def main(argv):
    if len(argv) == 3 and argv[1] == "--range":
        targets = commits_in_range(argv[2])
    elif len(argv) == 2:
        with open(argv[1], encoding="utf-8") as message_file:
            targets = [("(new)", message_file.read())]
    else:
        print(__doc__)
        return 2

    failed = False
    for commit_hash, message in targets:
        errors = check_message(message)
        if errors:
            failed = True
            print(f"✗ {commit_hash} {message.splitlines()[0] if message else ''}")
            for error in errors:
                print(f"    - {error}")

    if failed:
        print("\n커밋 메시지 규칙: GUIDE.md §4")
        return 1
    print(f"✓ 커밋 메시지 {len(targets)}개 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
