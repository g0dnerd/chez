#!/usr/bin/env fish

set PY_SCRIPT /home/paul/projects/chez/scripts/train.py
set UV_CMD uv run
set SHUTDOWN_CMD /usr/sbin/shutdown

# $SHUTDOWN_CMD --show
# $SHUTDOWN_CMD -c

echo "Starting: $UV_CMD $PY_SCRIPT"
$UV_CMD python $PY_SCRIPT --config large --iterations 5 --checkpoint models/iter_0010.pt
set EXIT_STATUS $status

$SHUTDOWN_CMD +2
