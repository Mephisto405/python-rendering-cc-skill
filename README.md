# Python Rendering Plugin for Claude Code

Python 3D 렌더링 라이브러리(nvdiffrast, PyTorch3D, pyrender, trimesh, Open3D) 사용을 위한 Claude Code 플러그인.

## Installation

### 방법 1: 마켓플레이스 사용 (권장)

```bash
# 1. 마켓플레이스 추가
/plugin marketplace add Mephisto405/python-rendering-cc-skill

# 2. 플러그인 설치
/plugin install python-rendering
```

### 방법 2: 로컬 설치

```bash
git clone https://github.com/Mephisto405/python-rendering-cc-skill.git
claude --plugin-dir ./python-rendering-cc-skill
```

### 방법 3: 수동 설치

```bash
# ~/.claude/skills/ 에 직접 복사
git clone https://github.com/Mephisto405/python-rendering-cc-skill.git
cp -r python-rendering-cc-skill/skills/python-rendering ~/.claude/skills/
```

## Usage

설치 후 Claude Code가 렌더링 관련 작업 시 자동으로 스킬을 참조합니다.

직접 호출:
```
/python-rendering
```

## Features

- **좌표계 컨벤션**: OpenGL, Blender, PyTorch3D, nvdiffrast 등 프레임워크별 축 컨벤션
- **렌더링 템플릿**: pyrender 5-point 조명, trimesh 검증된 코드 템플릿
- **텍스처 렌더링**: GLB/GLTF 텍스처 처리 방법
- **환경 설정**: Python 버전별 호환성, Windows/Linux 설정
- **문제 해결**: 일반적인 렌더링 오류 해결 방법
- **Anti-Aliasing**: 2x Supersampling + LANCZOS 다운샘플링

## Contents

```
python-rendering-cc-skill/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── skills/
│   └── python-rendering/
│       ├── SKILL.md               # 메인 스킬 (언어 라우터)
│       ├── guide-ko.md            # 한국어 가이드
│       ├── guide-en.md            # English guide
│       ├── axis-conventions-ko.md # 축 컨벤션 상세 (한국어)
│       └── axis-conventions-en.md # Axis conventions (English)
└── README.md
```

## License

MIT
