"""HTTP router layer.

Split from the ``api_server.py`` monolith as we extract each domain. New
endpoints should live in a submodule under ``api/routers/`` and be
included by ``api_server.py`` via ``app.include_router(...)``. The
monolith stays authoritative for state and glue during the migration - 
this package is where the surface migrates *to*, not another place it
also lives."""
