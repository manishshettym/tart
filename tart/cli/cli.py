import sys
import typer
import subprocess
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn

from tart.utils.tart_utils import print_header

app = typer.Typer(name="tart")


@app.command()
def init(
    gpu: bool = typer.Option(False, "--gpu", "-g", help="GPU version. default: False[cpu]"),
    cuda: str = typer.Option("cu117", "--cuda", "-c", help="CUDA version. default: cu116"),
):
    """Initialize tart learning environment based
    on your python, os, and hardware configs.
    """
    print_header()

    arch = cuda if gpu else "cpu"
    torch_link = f"https://download.pytorch.org/whl/nightly/{arch}"
    pyg_link = f"https://data.pyg.org/whl/torch-2.0.0+{arch}.html"
    tart = "\[tart]"
    success = "[bright_green]✔[/bright_green]"
    failed = "[bright_red]✘[/bright_red]"

    failed_install = False

    with Progress(
        SpinnerColumn(),
        TextColumn("Initializing tart learning env ...{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("", total=100)

        # install torch
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", torch_link, "--quiet"])
        except:
            progress.print(f"[bright_green] {tart} [/bright_green] {failed} torch")
            failed_install = True
        else:
            progress.print(f"[bright_green] {tart} [/bright_green] {success} torch")

        # install torch_geometric dependencies
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_scatter", "-f", pyg_link, "--quiet"])
        except:
            progress.print(f"[bright_green] {tart} [/bright_green] {failed} torch_scatter")
            failed_install = True
        else:
            progress.print(f"[bright_green] {tart} [/bright_green] {success} torch_scatter")

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_sparse", "-f", pyg_link, "--quiet"])
        except:
            progress.print(f"[bright_green] {tart} [/bright_green] {failed} torch_sparse")
            failed_install = True
        else:
            progress.print(f"[bright_green] {tart} [/bright_green] {success} torch_sparse")

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_cluster", "-f", pyg_link, "--quiet"])
        except:
            progress.print(f"[bright_green] {tart} [/bright_green] {failed} torch_cluster")
            failed_install = True
        else:
            progress.print(f"[bright_green] {tart} [/bright_green] {success} torch_cluster")

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_spline_conv", "-f", pyg_link, "--quiet"])
        except:
            progress.print(f"[bright_green] {tart} [/bright_green] {failed} torch_spline_conv")
            failed_install = True
        else:
            progress.print(f"[bright_green] {tart} [/bright_green] {success} torch_spline_conv")

        # install torch_geometric
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_geometric", "--quiet"])
        except:
            progress.print(f"[bright_green] {tart} [/bright_green] {failed} torch_geometric")
            failed_install = True
        else:
            progress.print(f"[bright_green] {tart} [/bright_green] {success} torch_geometric")

        # install deepsnap and transformers
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "deepsnap", "--quiet"])
        except:
            progress.print(f"[bright_green] {tart} [/bright_green] {failed} deepsnap")
            failed_install = True
        else:
            progress.print(f"[bright_green] {tart} [/bright_green] {success} deepsnap")

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "--quiet"])
        except:
            progress.print(f"[bright_green] {tart} [/bright_green] {failed} transformers")
            failed_install = True
        else:
            progress.print(f"[bright_green] {tart} [/bright_green] {success} transformers")

    rprint("\n")

    if not failed_install:
        rprint("🎉 You can now eat your [bright_green]tart[/bright_green] and have it! 🎉")
    else:
        rprint("🚨 Failed to initialize your [bright_green]tart[/bright_green] learning env. 🚨")

    return 0


@app.command()
def tart():
    """Tart is a CLI tool to help you manage your projects."""
    print_header()

    return 0


typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
