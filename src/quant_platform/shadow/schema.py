"""Shadow campaign schema constants, re-exported from the registry chain.

SF-S5-SL-MR4. The migration SQL lives in
:mod:`quant_platform.tracking.migrations` because the registry owns the schema
and its checksum-verified ledger. Defining it here and importing it there would
point the dependency outward, and it would also drag the shadow package into
the service container image, which ships only the tracking modules the service
actually runs -- the container source-inventory test catches exactly that.
"""

from __future__ import annotations

from typing import Final

from quant_platform.tracking.migrations import MIGRATIONS

#: Version of the shadow campaign schema within the registry migration chain.
SHADOW_SCHEMA_VERSION: Final = 2

SHADOW_MIGRATION_NAME: Final = "shadow_forecast_campaigns"

#: The migration SQL, sourced from the single definition in the registry chain
#: so the two can never drift apart.
SHADOW_MIGRATION_SQL: Final = next(
    migration.sql for migration in MIGRATIONS if migration.version == SHADOW_SCHEMA_VERSION
)

__all__ = [
    "SHADOW_MIGRATION_NAME",
    "SHADOW_MIGRATION_SQL",
    "SHADOW_SCHEMA_VERSION",
]
