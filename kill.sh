ps -eaf | grep asgi | grep -v grep | awk '{print $2}' | xargs kill -s 9
