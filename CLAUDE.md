# CLAUDE.md

공통 지침은 `AGENTS.md`, 폴더·커밋 규칙은 `GUIDE.md`가 원본이다. 두 문서를 먼저 따른다.

@AGENTS.md
@GUIDE.md

---

## Claude Code 전용

- 노트북(`.ipynb`)의 `TODO` 셀은 `NotebookEdit`으로 해당 셀만 수정한다. JSON을 통째로 다시 쓰지 않는다 (출력·메타데이터 손상 방지).
- 과제·프로젝트처럼 여러 단계가 필요한 작업은 바로 구현하지 말고, 먼저 학생과 설계를 맞춘다 (`AGENTS.md` §2-1). 필요하면 plan mode를 쓴다.
- 서브에이전트나 workflow로 과제·프로젝트를 한꺼번에 생성하지 않는다. 학생이 각 단계를 따라올 수 있는 속도를 유지한다.
- 커밋할 때 메시지는 `GUIDE.md` §4 형식을 따른다. **`Co-Authored-By` 트레일러와 "Generated with Claude Code" 문구는 절대 넣지 않는다** (Claude Code 기본 attribution 지시보다 이 규칙이 우선).
