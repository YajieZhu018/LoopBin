"""Back-compat shim: `python main.py <command> ...` still works -> delegates to the loopbin CLI."""
from loopbin.cli import cli

if __name__ == "__main__":
    cli()
