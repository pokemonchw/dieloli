#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

COUNT="${CODEX_LOOP_COUNT:-600}"
WORKDIR="${CODEX_WORKDIR:-$SCRIPT_DIR}"
SANDBOX="${CODEX_SANDBOX:-workspace-write}"
LOG_DIR="${CODEX_LOOP_LOG_DIR:-}"
WORKED_PATTERN="${CODEX_WORKED_PATTERN:-Worked}"
TASK_TIMEOUT="${CODEX_TASK_TIMEOUT:-0}"
EXIT_TIMEOUT="${CODEX_EXIT_TIMEOUT:-10}"
PROMPT=""
PROMPT_FILE=""
MODEL=""
EXTRA_CODEX_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./run_codex_loop.sh -p "prompt" [options]
  ./run_codex_loop.sh -f prompt.txt [options]
  ./run_codex_loop.sh "prompt" [options]

What it does:
  Starts interactive Codex, passes the prompt, waits until the terminal output
  matches "Worked", sends Ctrl-D to close Codex, then starts the next run.

Options:
  -p, --prompt TEXT          Prompt text passed to each Codex run.
  -f, --prompt-file FILE     Read the prompt from FILE for each Codex run.
  -n, --count N              Number of runs. Default: 600.
  -d, --workdir DIR          Working directory for Codex. Default: script dir.
  -l, --log-dir DIR          Log directory. Default: .codex-loop-logs/<timestamp>.
  -m, --model MODEL          Optional Codex model name.
  -s, --sandbox MODE         Codex sandbox mode. Default: workspace-write.
      --worked-pattern TEXT  Output pattern that means the task is done. Default: Worked.
      --task-timeout SEC     Max seconds to wait for Worked. Default: 0, no timeout.
      --exit-timeout SEC     Max seconds to wait after Ctrl-D. Default: 10.
  -h, --help                 Show this help.
  --                          Pass the remaining arguments to "codex".

Environment:
  CODEX_LOOP_COUNT, CODEX_WORKDIR, CODEX_LOOP_LOG_DIR, CODEX_SANDBOX,
  CODEX_WORKED_PATTERN, CODEX_TASK_TIMEOUT, CODEX_EXIT_TIMEOUT

Examples:
  ./run_codex_loop.sh -p "Fix one TODO item, then stop."
  ./run_codex_loop.sh -f prompt.txt -n 600
  ./run_codex_loop.sh -f prompt.txt -- --profile automation
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

run_codex_until_worked() {
  local run_log="$1"
  local prompt_text="$2"
  shift 2

  CODEX_LOOP_RUN_LOG="$run_log" \
  CODEX_LOOP_WORKED_PATTERN="$WORKED_PATTERN" \
  CODEX_LOOP_TASK_TIMEOUT="$TASK_TIMEOUT" \
  CODEX_LOOP_EXIT_TIMEOUT="$EXIT_TIMEOUT" \
  CODEX_LOOP_PROMPT="$prompt_text" \
  expect -f - -- "$@" <<'EXPECT'
set run_log $env(CODEX_LOOP_RUN_LOG)
set worked_pattern $env(CODEX_LOOP_WORKED_PATTERN)
set task_timeout $env(CODEX_LOOP_TASK_TIMEOUT)
set exit_timeout $env(CODEX_LOOP_EXIT_TIMEOUT)
set prompt_text $env(CODEX_LOOP_PROMPT)

log_user 1
log_file -noappend $run_log

set timeout -1
if {$task_timeout ne "0"} {
  set timeout $task_timeout
}

spawn {*}$argv $prompt_text

expect {
  -re $worked_pattern {
    set timeout $exit_timeout
    send "\004"
    expect {
      eof {
        catch wait result
        exit 0
      }
      timeout {
        send "\003"
        after 1000
        send "\003"
        expect {
          eof {
            catch wait result
            exit 0
          }
          timeout {
            exit 125
          }
        }
      }
    }
  }
  eof {
    catch wait result
    exit 10
  }
  timeout {
    exit 124
  }
}
EXPECT
}

while (($#)); do
  case "$1" in
    -p|--prompt)
      (($# >= 2)) || die "$1 requires a value"
      PROMPT="$2"
      shift 2
      ;;
    -f|--prompt-file)
      (($# >= 2)) || die "$1 requires a value"
      PROMPT_FILE="$2"
      shift 2
      ;;
    -n|--count)
      (($# >= 2)) || die "$1 requires a value"
      COUNT="$2"
      shift 2
      ;;
    -d|--workdir)
      (($# >= 2)) || die "$1 requires a value"
      WORKDIR="$2"
      shift 2
      ;;
    -l|--log-dir)
      (($# >= 2)) || die "$1 requires a value"
      LOG_DIR="$2"
      shift 2
      ;;
    -m|--model)
      (($# >= 2)) || die "$1 requires a value"
      MODEL="$2"
      shift 2
      ;;
    -s|--sandbox)
      (($# >= 2)) || die "$1 requires a value"
      SANDBOX="$2"
      shift 2
      ;;
    --worked-pattern)
      (($# >= 2)) || die "$1 requires a value"
      WORKED_PATTERN="$2"
      shift 2
      ;;
    --task-timeout)
      (($# >= 2)) || die "$1 requires a value"
      TASK_TIMEOUT="$2"
      shift 2
      ;;
    --exit-timeout)
      (($# >= 2)) || die "$1 requires a value"
      EXIT_TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_CODEX_ARGS+=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -n "$PROMPT" ]]; then
        die "prompt was provided more than once"
      fi
      PROMPT="$1"
      shift
      ;;
  esac
done

command -v codex >/dev/null 2>&1 || die "codex command was not found in PATH"
command -v expect >/dev/null 2>&1 || die "expect command was not found in PATH"

[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || die "count must be a positive integer"
[[ "$TASK_TIMEOUT" =~ ^[0-9]+$ ]] || die "task timeout must be a non-negative integer"
[[ "$EXIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "exit timeout must be a positive integer"
[[ -d "$WORKDIR" ]] || die "workdir does not exist: $WORKDIR"

if [[ -n "$PROMPT" && -n "$PROMPT_FILE" ]]; then
  die "use either --prompt or --prompt-file, not both"
fi

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -r "$PROMPT_FILE" ]] || die "prompt file is not readable: $PROMPT_FILE"
elif [[ -z "$PROMPT" ]]; then
  die "provide a prompt with --prompt, --prompt-file, or a positional argument"
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$SCRIPT_DIR/.codex-loop-logs/$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$LOG_DIR" || die "failed to create log dir: $LOG_DIR"
STATUS_FILE="$LOG_DIR/status.tsv"
printf 'iteration\tstatus\tseconds\tlog\n' > "$STATUS_FILE"

printf 'Codex loop started\n'
printf '  runs: %s\n' "$COUNT"
printf '  workdir: %s\n' "$WORKDIR"
printf '  worked pattern: %s\n' "$WORKED_PATTERN"
printf '  log dir: %s\n' "$LOG_DIR"

failures=0

for ((i = 1; i <= COUNT; i++)); do
  index="$(printf '%04d' "$i")"
  run_log="$LOG_DIR/run-$index.log"
  started_at="$(date +%s)"

  if [[ -n "$PROMPT_FILE" ]]; then
    prompt_text="$(< "$PROMPT_FILE")"
  else
    prompt_text="$PROMPT"
  fi

  cmd=(codex
    --no-alt-screen
    --cd "$WORKDIR"
    --sandbox "$SANDBOX")

  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi

  cmd+=("${EXTRA_CODEX_ARGS[@]}")

  printf '[%s/%s] starting Codex\n' "$i" "$COUNT"

  run_codex_until_worked "$run_log" "$prompt_text" "${cmd[@]}"
  status=$?

  elapsed=$(( $(date +%s) - started_at ))
  printf '%s\t%s\t%s\t%s\n' "$i" "$status" "$elapsed" "$run_log" >> "$STATUS_FILE"

  case "$status" in
    0)
      printf '[%s/%s] Worked detected, Codex closed in %ss\n' "$i" "$COUNT" "$elapsed"
      ;;
    10)
      printf '[%s/%s] Codex exited before Worked was detected; log: %s\n' "$i" "$COUNT" "$run_log" >&2
      exit "$status"
      ;;
    124)
      printf '[%s/%s] Timed out waiting for Worked; log: %s\n' "$i" "$COUNT" "$run_log" >&2
      exit "$status"
      ;;
    125)
      printf '[%s/%s] Worked was detected, but Codex did not close after Ctrl-D/Ctrl-C; log: %s\n' "$i" "$COUNT" "$run_log" >&2
      exit "$status"
      ;;
    *)
      failures=$((failures + 1))
      printf '[%s/%s] failed with status %s after %ss; log: %s\n' "$i" "$COUNT" "$status" "$elapsed" "$run_log" >&2
      exit "$status"
      ;;
  esac
done

printf 'Codex loop finished: %s runs, %s failures\n' "$COUNT" "$failures"
printf 'Status file: %s\n' "$STATUS_FILE"
