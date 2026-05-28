#!/bin/bash
# Always rebuild the image so viewer/dist + orbit_wars_app/ pick up local
# edits — plain `docker compose up` reuses the cached image and silently
# serves stale code.
set -e
docker compose up --build "$@"
