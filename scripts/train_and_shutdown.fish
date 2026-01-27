#!/usr/bin/env fish

set PY_SCRIPT /home/paul/projects/chez/scripts/train.py
set UV_CMD uv run
set SHUTDOWN_CMD /usr/sbin/shutdown

echo "Starting: $UV_CMD $PY_SCRIPT"
$UV_CMD python $PY_SCRIPT --config medium --iterations 10
set EXIT_STATUS $status

$SHUTDOWN_CMD +2
