#!/usr/bin/env bash
# Creates a StudioLite launcher entry on Linux (XDG .desktop file).
# Drops it in both ~/.local/share/applications (so it shows up in the app
# launcher / Activities) and ~/Desktop (so you have a clickable icon).
# Marks launch.sh executable. Re-running overwrites cleanly.

set -eu

NAME="${1:-StudioLite}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="$ROOT/launch.sh"
ICON_SRC="$ROOT/web/app/favicon.ico"

if [[ ! -f "$LAUNCH" ]]; then
    echo "launch.sh not found at $LAUNCH" >&2
    exit 1
fi

chmod +x "$LAUNCH"

# Resolve a terminal emulator to wrap launch.sh in (.desktop with
# Terminal=true relies on a system-default terminal, which is unreliable
# across distros). We pick the first one available.
TERM_CMD=""
for t in x-terminal-emulator gnome-terminal konsole xfce4-terminal mate-terminal tilix alacritty kitty xterm; do
    if command -v "$t" >/dev/null 2>&1; then
        TERM_CMD="$t"
        break
    fi
done

# Compose the Exec= line. Each terminal has its own "run this command" flag.
case "$TERM_CMD" in
    gnome-terminal)         EXEC_LINE="gnome-terminal -- bash -c '\"$LAUNCH\"; exec bash'" ;;
    konsole)                EXEC_LINE="konsole -e bash -c '\"$LAUNCH\"; exec bash'" ;;
    xfce4-terminal)         EXEC_LINE="xfce4-terminal --hold -e \"bash -c '$LAUNCH'\"" ;;
    mate-terminal)          EXEC_LINE="mate-terminal -- bash -c '\"$LAUNCH\"; exec bash'" ;;
    tilix)                  EXEC_LINE="tilix -e bash -c '\"$LAUNCH\"; exec bash'" ;;
    alacritty|kitty|xterm)  EXEC_LINE="$TERM_CMD -e bash -c '\"$LAUNCH\"; exec bash'" ;;
    x-terminal-emulator)    EXEC_LINE="x-terminal-emulator -e bash -c '\"$LAUNCH\"; exec bash'" ;;
    "")                     EXEC_LINE="\"$LAUNCH\""; echo "Note: no terminal emulator found; .desktop will run launch.sh directly (no window)." ;;
esac

# Build an icon path - copy favicon.ico into a stable location and use it,
# or just point at the source file. Most desktops accept .ico via gdk-pixbuf.
ICON=""
if [[ -f "$ICON_SRC" ]]; then
    ICON_DIR="$HOME/.local/share/icons"
    mkdir -p "$ICON_DIR"
    cp -f "$ICON_SRC" "$ICON_DIR/studiolite.ico"
    ICON="$ICON_DIR/studiolite.ico"
fi

APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"
DESKTOP_FILE="$APPS_DIR/$NAME.desktop"

{
    echo "[Desktop Entry]"
    echo "Type=Application"
    echo "Version=1.0"
    echo "Name=$NAME"
    echo "Comment=Launch StudioLite (FastAPI + Next.js) and open the web UI"
    echo "Exec=$EXEC_LINE"
    echo "Path=$ROOT"
    [[ -n "$ICON" ]] && echo "Icon=$ICON"
    echo "Terminal=false"
    echo "Categories=Development;AudioVideo;"
    echo "StartupNotify=true"
} > "$DESKTOP_FILE"
chmod +x "$DESKTOP_FILE"

# Copy to Desktop (XDG-compliant fallback to ~/Desktop)
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
mkdir -p "$DESKTOP_DIR"
cp -f "$DESKTOP_FILE" "$DESKTOP_DIR/$NAME.desktop"
chmod +x "$DESKTOP_DIR/$NAME.desktop"

# GNOME requires the desktop file to be "trusted" - newer versions use this
# metadata attribute. Older GNOME wants it marked executable (done above).
if command -v gio >/dev/null 2>&1; then
    gio set "$DESKTOP_DIR/$NAME.desktop" metadata::trusted true 2>/dev/null || true
fi

# Refresh the application database so the launcher entry shows up
# immediately.
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi

echo
echo "Launcher installed:"
echo "  Apps    : $DESKTOP_FILE"
echo "  Desktop : $DESKTOP_DIR/$NAME.desktop"
[[ -n "$TERM_CMD" ]] && echo "  Terminal: $TERM_CMD"
echo
echo "To pin to your dock/taskbar:"
echo "  GNOME : open Activities, search for 'StudioLite', right-click -> 'Add to Favorites'"
echo "  KDE   : right-click the launcher icon in the menu -> 'Pin to Task Manager'"
echo "  Unity : drag the launcher icon onto the dock"
echo
