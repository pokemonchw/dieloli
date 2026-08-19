---
name: dieloli-project
description: Follow DieLoli project conventions when modifying this repository. Use for coding, refactoring, data/config changes, UI panel work, behavior/premise/state-machine extensions, localization, build, or validation tasks in the DieLoli Python game project.
---

# DieLoli Project Conventions

## Project Shape

Work as if this is a data-driven Python game, not a conventional web/service app.

- `game.py` is the runtime entrypoint. It initializes normal config, cache, translation, generated game config, name/map data, feature modules, AI server, then starts the UI flow.
- `Script/Core/` contains engine-level utilities, cache globals, type containers, save/io, drawing primitives, path/config helpers, and translation helpers.
- `Script/Config/` contains generated and loaded configuration. `config_def.py` is generated from CSV metadata; `game_config.py` loads `data/data.json` into global config maps.
- `Script/Design/` contains game logic: character behavior, time, weather, instruction handling, settling, state machines, events, achievements, AI integration, and constants.
- `Script/Premise/`, `Script/Settle/`, and `Script/StateMachine/` are extension modules registered by decorators and imported for side effects.
- `Script/UI/Model/`, `Script/UI/Panel/`, and `Script/UI/Flow/` implement text/ASCII-style UI drawing and panel flows.
- `tools/` contains PySide6 editors and CSV/build helpers. Treat these as separate tooling apps with their own local imports.
- `data/` is source/runtime data: CSV tables, generated `data.json`, maps, events, clothing, clubs, translation files, fonts, and model/cache files.

## Coding Style

Match the existing Python style unless a local file clearly uses a different pattern.

- Use Python 3 syntax compatible with the CI target, currently Python 3.13 in GitHub Actions.
- Keep imports at the top, grouped roughly as standard library, third-party, then `Script.*` imports. Do not introduce broad formatting-only churn.
- Use 4-space indentation.
- Prefer `snake_case` for functions, variables, module globals, and generated handler names.
- Use `PascalCase` for classes and constant-container classes.
- Constants live in `Script/Design/constant/*.py` as class attributes with uppercase names and Chinese doc comments.
- Existing data container classes are plain classes with annotated attributes and many `""" 中文说明 """` attribute docstrings. Follow that style rather than introducing dataclasses unless the surrounding module already uses them.
- Keep type hints lightweight and local: `cache: game_type.Cache`, `character_data: game_type.Character`, `return_list: List[str]`, etc.
- Project comments and docstrings are primarily Chinese. Add concise Chinese docstrings/comments for new public functions, data fields, handlers, panels, and generated constants.
- Existing function docstrings often use:
  ```python
  """
  简短说明
  Keyword arguments:
  name -- 参数说明
  Return arguments:
  type -- 返回说明
  """
  ```
  Preserve this style when adding similar functions.
- Use `_()` from `Script.Core.get_text` for player-facing strings and menu labels. Keep internal keys, IDs, and file paths untranslated.

## Runtime State

Respect the global cache/config architecture.

- Access runtime state through `cache_control.cache`, normally assigned to a module global:
  ```python
  cache: game_type.Cache = cache_control.cache
  """ 游戏缓存数据 """
  ```
- Access config through `Script.Config.game_config` globals such as `config_character_state`, `config_school_session_data`, and `config_event`.
- Keep player character assumptions consistent: character id `0` is the player in most gameplay logic.
- Check target relationships defensively. Many premise functions return `0` when `target_character_id == -1`.
- Mutate `game_type` objects directly when following existing behavior code; do not introduce a separate state manager.
- Use existing helpers for cross-cutting behavior: `character`, `attr_calculation`, `game_time`, `map_handle`, `course`, `clothing`, `update`, `text_handle`, and `draw` rather than reimplementing their logic.

## Extension Patterns

Use the existing decorator registration mechanisms.

- Instructions:
  - Add/extend constants in `Script/Design/constant/instruct.py` and related type/panel constants when needed.
  - Register handlers with `@handle_instruct.add_instruct(...)`.
  - Put handlers in the matching `Script/Design/instruct/*.py` module.
  - Ensure the module is imported by `Script/Design/instruct/__init__.py` if it is new.
- Premises:
  - Add constants in `Script/Design/constant/premise.py`.
  - Register checks with `@handle_premise.add_premise(constant.Premise.X)`.
  - Put related checks in the matching `Script/Premise/*.py` module.
  - Ensure a new premise module is imported in `Script/Premise/__init__.py`.
  - Return `1` for pass and `0` for fail unless the surrounding premise category uses weighted values.
- Settle behavior:
  - Register effects with `@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.X)`.
  - Handler signature should match:
    `character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int`.
  - Update both character state and `change_data` when the player should see a settlement change.
- State machines:
  - Register with `@handle_state_machine.add_state_machine(constant.StateMachine.X)`.
  - Put behavior assignment, duration, state, target/course fields, and other runtime mutations in the handler.
  - Ensure new modules are imported by `Script/StateMachine/__init__.py`.
- UI panels:
  - Build panel objects around `draw.*` and `panel.*` model classes.
  - Keep `return_list` updated for interactive panels.
  - Use `normal_config.config_normal.text_width` or constructor `width` instead of hardcoded screen width when possible.

## Data And Generated Files

Prefer source data changes over generated output changes.

- CSV source tables live in `data/csv/`. Their first rows define documentation, types, gettext flags, and class description for generation.
- `buildconfig.py` generates at least:
  - `Script/Config/config_def.py`
  - `data/data.json`
  - `package.json`
  - translation PO updates on Linux
- Do not hand-edit `Script/Config/config_def.py` or `data/data.json` for durable changes. Edit the source CSV/JSON and regenerate.
- Event, target, club, and clothing data are sourced from JSON under `data/event/`, `data/target/`, `data/club/`, and `data/clothing/`, with editor defaults under `tools/Dieloli*Editor/default.json`.
- Build helpers may copy editor defaults into `data/`; avoid deleting or renaming those defaults without updating build scripts and editors.
- Some helper scripts in `tools/` generate repetitive behavior code from CSV. Prefer updating the CSV/template path if a change is systemic.

## Localization

Preserve gettext behavior.

- Wrap player-visible strings in `_()` where the surrounding module does so.
- Initialize `_` with:
  ```python
  _: FunctionType = get_text._
  """ 翻译api """
  ```
- Config strings marked by CSV gettext flags are extracted by `buildconfig.py`; code strings are extracted by `buildpo.py`.
- Translation files live under `data/po/<language>/LC_MESSAGES/dieloli.po` and compiled `.mo` files are built by `buildmo.py`.

## Tools And Editors

When working in `tools/Dieloli*Editor/`:

- Follow PySide6 patterns already present in that editor.
- Tool modules use local imports such as `from ui.main_tabs import MainTabs`; do not rewrite them into `Script.*` imports unless migrating the tool intentionally.
- Keep editor JSON read/write behavior compatible with the existing default files.

## Validation

There is no dedicated test suite in this repository. Validate with the narrowest useful checks for the change.

- For syntax/import sanity on touched Python files, run:
  ```bash
  python -m compileall <paths>
  ```
- For config/data changes, run:
  ```bash
  python buildconfig.py
  python init_data.py
  ```
- For translation changes, run when gettext tools are available:
  ```bash
  python buildpo.py
  python buildmo.py
  ```
- For full runtime smoke testing, run:
  ```bash
  python game.py
  ```
  This starts the interactive game UI and may be unsuitable for automated sessions.
- CI builds install `requirements.txt`, run `buildconfig.py`, run `init_data.py`, and package `game.py` with PyInstaller on Windows/macOS.

## Change Discipline

- Keep changes narrow. This project has many generated constants, global registries, and side-effect imports; unrelated refactors can silently change runtime registration.
- Before editing a feature, inspect neighboring modules in the same category and copy their registration, naming, docstring, and cache-access patterns.
- Do not replace global cache/config access with dependency injection in isolated changes.
- Do not introduce new frameworks, formatters, or lint rules unless explicitly requested.
- Avoid broad auto-formatting. Existing files contain long lines and locally inconsistent spacing; only format code you materially change.
- When adding a new module that relies on decorator registration, verify it is imported by the relevant package `__init__.py`; otherwise the feature will not register at runtime.
