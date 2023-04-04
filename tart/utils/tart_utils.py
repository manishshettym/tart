from rich.console import Console

console = Console()


def print_header():
    header = """ _                 _   
| |_   __ _  _ __ | |_ 
| __| / _` || '__|| __|
| |_ | (_| || |   | |_ 
 \__| \__,_||_|    \__|"""
    
    console.print(f"[bright_green]{header}[/bright_green]\n")


def summarize_tart_run(args):
    tart = '\[tart]'
    console.print(f"[bright_green] {tart} [/bright_green] dataset {args.dataset}")
    console.print(f"[bright_green] {tart} [/bright_green] n_iters {args.n_iters}")
    console.print(f"[bright_green] {tart} [/bright_green] n_batches {args.n_batches}")
    console.print(f"[bright_green] {tart} [/bright_green] batch_size {args.batch_size}")
    console.print(f"[bright_green] {tart} [/bright_green] n_train {args.n_batches} x {args.batch_size} = {args.n_train}")
    console.print(f"[bright_green] {tart} [/bright_green] n_test 0.2x{args.n_train} = {args.n_test}")
