"""VISTA — VISualization of relation Topologies of Alternatives in MCDA methods."""

__version__ = "0.1.0"

from mcda_vista.core import VistaGenerator, VistaResult, generate_vista
from mcda_vista.plotting import plot_vista, plot_vista_comparison, plot_vista_grid
from mcda_vista.relation import Relation

__all__ = [
    "generate_vista",
    "plot_vista",
    "plot_vista_comparison",
    "plot_vista_grid",
    "Relation",
    "VistaGenerator",
    "VistaResult",
]
