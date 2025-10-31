"""Utilities to build and import state-level electricity LCIs into Brightway.

This module provides a thin orchestration layer around the existing
ElectricityLCI pipeline so that users can programmatically build a
state-specific life cycle inventory, import it into a Brightway25 project,
and calculate TRACI v2.1 impact scores for a selected set of categories.

The workflow exposed here intentionally mirrors the three high-level steps a
practitioner would carry out manually:

1. Configure and run the ElectricityLCI generation routines with the
   ``STATE`` aggregation option to produce the technology- and mix-level
   processes for each U.S. state.
2. Import the resulting JSON-LD archive into a Brightway project using the
   OLCA JSON importer, optionally installing the TRACI v2.1 LCIA package if it
   is not already available.
3. Evaluate the TRACI v2.1 impact scores for any state-level generation mix
   process of interest.

Example
-------
>>> from electricitylci.state_brightway import (
...     generate_state_inventory,
...     import_state_inventory_to_brightway,
...     compute_traci_impacts,
... )
>>> jsonld_path = generate_state_inventory()["jsonld"]
>>> import_state_inventory_to_brightway(jsonld_path)
>>> compute_traci_impacts(
...     database="ELCI_STATE", region="CA",
...     methods=[("TRACI 2.1", "Total", "global warming")]
... )
{"('TRACI 2.1', 'Total', 'global warming')": 7.3}

Notes
-----
The helper functions assume that the ``ELCI_STATE`` model configuration file
is present (added in this change set) and that the caller has supplied the
necessary API keys for external data downloads in that configuration.  The
Brightway-related utilities depend on :mod:`bw2data`, :mod:`bw2io`, and
:mod:`bw2calc`; informative error messages are raised if these packages are
unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import electricitylci.model_config as config
from electricitylci import (
    get_generation_mix_process_df,
    run_post_processes,
    write_generation_mix_database_to_dict,
    write_process_dicts_to_jsonld,
)
from electricitylci.main import run_generation
from electricitylci.utils import get_logger, rollover_logger

__all__ = [
    "compute_traci_impacts",
    "generate_state_inventory",
    "import_state_inventory_to_brightway",
]


def generate_state_inventory(
    model_name: str = "ELCI_STATE",
    *,
    log_to_console: bool = True,
    run_post_processing: bool = False,
) -> Dict[str, object]:
    """Build the state-aggregated generation and mix processes.

    Parameters
    ----------
    model_name:
        Name of the ElectricityLCI configuration to use.  The helper expects a
        configuration with ``regional_aggregation: 'STATE'``; the default is the
        configuration added by this change set (``ELCI_STATE``).
    log_to_console:
        When ``True`` (default) a stream logger is activated so that
        ElectricityLCI progress messages are printed to stdout.  Set to
        ``False`` to rely on the rotating file handler only.
    run_post_processing:
        Optional flag to invoke :func:`electricitylci.run_post_processes` after
        the generation and mix processes have been written to JSON-LD.  The
        default ``False`` avoids deleting or altering intermediate files that
        may be inspected downstream.

    Returns
    -------
    dict
        A dictionary containing three keys:

        ``jsonld``
            Path to the JSON-LD archive containing the state-level unit
            processes.  The archive name is derived from the ElectricityLCI
            configuration and timestamp and matches
            :attr:`model_specs.namestr`.
        ``generation``
            The dictionary returned by :func:`electricitylci.main.run_generation`
            for potential downstream inspection.
        ``generation_mix``
            A dictionary describing the state-level generation mix processes as
            produced by :func:`electricitylci.write_generation_mix_database_to_dict`.
    """
    log = get_logger(stream=log_to_console, rfh=True)
    config.model_specs = config.build_model_class(model_name)
    logging.info("Building state-level generation inventory")

    try:
        generation_dict = run_generation()
        logging.info("Deriving state generation mix processes")
        generation_mix_df = get_generation_mix_process_df("STATE")
        generation_mix_dict = write_generation_mix_database_to_dict(
            generation_mix_df, generation_dict
        )
        write_process_dicts_to_jsonld(generation_mix_dict)

        if run_post_processing and config.model_specs.run_post_processes:
            run_post_processes()

        result = {
            "jsonld": config.model_specs.namestr,
            "generation": generation_dict,
            "generation_mix": generation_mix_dict,
        }
    finally:
        rollover_logger(log)

    return result


def import_state_inventory_to_brightway(
    jsonld_path: Path | str,
    *,
    project_name: str = "ElectricityLCI-State",
    database_name: Optional[str] = None,
    overwrite: bool = False,
    install_traci: bool = True,
    traci_package: str = "TRACI 2.1",
) -> Dict[str, object]:
    """Import the state-level inventory into a Brightway25 project.

    Parameters
    ----------
    jsonld_path:
        Path to the JSON-LD archive produced by
        :func:`generate_state_inventory`.
    project_name:
        Name of the Brightway project that will receive the database.  The
        project is created if it does not already exist.  Defaults to
        ``"ElectricityLCI-State"``.
    database_name:
        Optional custom name for the Brightway database.  When omitted, the
        stem of ``jsonld_path`` is used.
    overwrite:
        If ``True`` and a database with the selected name already exists, it is
        deleted prior to the import.  The default ``False`` preserves existing
        data and raises a :class:`ValueError` when a name collision occurs.
    install_traci:
        Install the specified TRACI LCIA package (defaults to ``"TRACI 2.1"``)
        using :mod:`bw2io` when the method group is not present.  Set to
        ``False`` to skip this step.
    traci_package:
        Name of the LCIA package to install via :class:`bw2io.importers.lciafmt`
        when ``install_traci`` is ``True``.

    Returns
    -------
    dict
        Keys include ``project`` (the active project name), ``database`` (the
        Brightway database that was created), and ``statistics`` containing the
        summary returned by :meth:`bw2io.importers.base_importer.BaseImporter.statistics`.
    """
    jsonld_path = Path(jsonld_path)
    if database_name is None:
        database_name = jsonld_path.stem

    try:
        from bw2data import Database, databases, projects
        from bw2io.importers.olca import OLCAJsonImporter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Brightway25 packages (bw2data and bw2io) are required to import the "
            "ElectricityLCI archive."
        ) from exc

    projects.set_current(project_name)

    if database_name in databases:
        if not overwrite:
            raise ValueError(
                f"A Brightway database named '{database_name}' already exists. "
                "Set overwrite=True to replace it."
            )
        Database(database_name).delete()

    importer = OLCAJsonImporter(str(jsonld_path), database_name)
    importer.apply_strategies()
    stats = importer.statistics()
    importer.write_database()

    if install_traci:
        try:
            _ensure_traci_package(traci_package)
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning(
                "Unable to install LCIA package '%s': %s", traci_package, exc
            )

    return {"project": project_name, "database": database_name, "statistics": stats}


def compute_traci_impacts(
    database: str,
    region: str,
    methods: Sequence[Tuple[str, ...]],
    *,
    demand_multiplier: float = 1.0,
) -> Dict[str, float]:
    """Evaluate TRACI scores for a state-level generation mix.

    Parameters
    ----------
    database:
        Name of the Brightway database that contains the imported ElectricityLCI
        processes.
    region:
        Two-letter U.S. state or District of Columbia abbreviation.  The helper
        looks for a process named ``Electricity; at grid; generation mix -
        <region>`` within the selected database.
    methods:
        Iterable of Brightway LCIA method tuples (e.g.,
        ``("TRACI 2.1", "Total", "global warming")``).  Methods must already be
        registered in Brightway (use ``install_traci=True`` in
        :func:`import_state_inventory_to_brightway`).
    demand_multiplier:
        Scalar applied to the functional unit when building the Brightway LCA
        model.  Defaults to ``1.0`` for a single megawatt-hour.

    Returns
    -------
    dict
        Mapping of the serialized method tuple (as a string) to the calculated
        LCIA score.
    """
    try:
        from bw2calc import LCA
        from bw2data import Database
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Brightway25 packages (bw2data and bw2calc) are required to compute impacts."
        ) from exc

    db = Database(database)
    target_name = f"Electricity; at grid; generation mix - {region.upper()}"

    try:
        activity = next(act for act in db if act["name"] == target_name)
    except StopIteration as exc:
        raise ValueError(
            f"No state-level generation mix named '{target_name}' was found in the "
            f"'{database}' database."
        ) from exc

    demand = {activity: demand_multiplier}
    results: Dict[str, float] = {}

    for method in methods:
        lca = LCA(demand, method)
        lca.lci()
        lca.lcia()
        results[str(method)] = float(lca.score)

    return results


def _ensure_traci_package(package: str) -> None:
    """Install the requested TRACI LCIA package if it is missing.

    The helper uses :class:`bw2io.importers.lciafmt.LCIAImporter` to install
    LCIA data distributed through ``lciafmt``.  When the package is already
    present the function returns immediately.
    """
    try:
        from bw2data import methods
        from bw2io.importers.lciafmt import LCIAImporter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Installing TRACI packages requires the bw2data, bw2io, and lciafmt packages."
        ) from exc

    if any(package in method[0] for method in methods):
        return

    importer = LCIAImporter([package])
    importer.apply_strategies()
    importer.write_methods()
