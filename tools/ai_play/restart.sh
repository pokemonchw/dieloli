PATHS=("../../")
for i in "${!PATHS[@]}"; do
    cd "${PATHS[$i]}" || continue
    nohup ./game.py >nohup.out 2>&1 &
    cd - >/dev/null
done

