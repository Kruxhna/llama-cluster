"""
llama-cluster (AeroMesh): Distributed LLM inference cluster orchestrator.
"""

__version__ = "0.1.0"
__author__ = "Krushna"


def main() -> None:
    from llama_cluster.cli import main as cli_main
    cli_main()
