# Always rebuild the image so viewer/dist + orbit_wars_app/ pick up local
# edits. Plain `docker compose up` may reuse the cached image.

$ErrorActionPreference = "Stop"

docker compose up --build @args