"""VISTA — VISualization of relation Topologies of Alternatives."""

__version__ = "0.2.0"

from mcda_vista.core import VistaGenerator, VistaResult, generate_vista
from mcda_vista.plotting import plot_vista, plot_vista_comparison, plot_vista_grid
from mcda_vista.protocol import (
    CheckResult,
    ProtocolReport,
    SensitivityRow,
    plot_protocol_report,
    run_protocol,
)
from mcda_vista.relation import Relation

__all__ = [
    "CheckResult",
    "generate_vista",
    "plot_protocol_report",
    "plot_vista",
    "plot_vista_comparison",
    "plot_vista_grid",
    "ProtocolReport",
    "Relation",
    "run_protocol",
    "SensitivityRow",
    "VistaGenerator",
    "VistaResult",
]
