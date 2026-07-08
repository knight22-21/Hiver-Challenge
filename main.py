"""
main.py — End-to-end demo runner

Usage:
  python main.py                          # evaluate on 5 held-out dataset emails
  python main.py --provider groq          # use Groq (Llama 3.3 70B) instead of Claude
  python main.py --custom                 # generate reply for a hand-typed email
  python main.py --id 3                  # test a specific email by dataset ID
  python main.py --all                    # evaluate every email in the dataset
  python main.py --provider groq --all   # full run with Groq
"""

import argparse
import json
import sys
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from generator import EmailResponseGenerator
from evaluator import evaluate

console = Console()


def fmt_score(val: float, out_of: float = 1.0) -> str:
    pct = val / out_of
    color = "green" if pct >= 0.7 else ("yellow" if pct >= 0.5 else "red")
    return f"[{color}]{val:.3f}[/{color}]"


def run_evaluation(email: dict, generator: EmailResponseGenerator, idx: int, total: int):
    console.rule(f"[bold cyan][{idx}/{total}] ID={email['id']} | {email['category'].upper()}[/bold cyan]")
    console.print(f"[bold]Subject:[/bold] {email['subject']}")
    console.print()

    # Shorten email for display
    snippet = email["incoming_email"][:220].replace("\n", " ")
    if len(email["incoming_email"]) > 220:
        snippet += "…"
    console.print(Panel(snippet, title="Incoming Email (excerpt)", border_style="blue"))

    # Generate
    with console.status("[cyan]Generating reply…[/cyan]"):
        gen = generator.generate(email["incoming_email"], email["subject"])

    reply_snippet = gen["generated_reply"][:400]
    if len(gen["generated_reply"]) > 400:
        reply_snippet += "…"
    console.print(Panel(reply_snippet, title="Generated Reply (excerpt)", border_style="green"))

    retrieved = ", ".join(f"#{e['id']}({e['category']})" for e in gen["retrieved_examples"])
    console.print(f"  [dim]Retrieved examples: {retrieved}[/dim]")
    console.print()

    # Evaluate
    with console.status("[cyan]Evaluating…[/cyan]"):
        scores = evaluate(
            incoming_email=email["incoming_email"],
            generated_reply=gen["generated_reply"],
            reference_reply=email.get("reply"),
        )

    # Build score table
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Notes")

    judge = scores.get("llm_judge", {})
    dims = ["relevance", "tone", "completeness", "accuracy", "clarity"]
    for dim in dims:
        d = judge.get(dim, {})
        if isinstance(d, dict):
            s = d.get("score", "—")
            reason = d.get("reason", "")[:80]
            table.add_row(
                f"  {dim.capitalize()}",
                fmt_score(s, 10),
                f"[dim]{reason}[/dim]",
            )

    table.add_row("", "", "")
    table.add_row("LLM Judge (avg)", fmt_score(scores.get("llm_judge_avg", 0), 10), "Mean of 5 dimensions")

    if "rouge" in scores:
        r = scores["rouge"]
        table.add_row("ROUGE-1", fmt_score(r["rouge1"]), "Unigram overlap w/ reference")
        table.add_row("ROUGE-2", fmt_score(r["rouge2"]), "Bigram overlap w/ reference")
        table.add_row("ROUGE-L", fmt_score(r["rougeL"]), "Longest common subsequence")
        table.add_row(
            "Semantic Similarity",
            fmt_score(scores.get("semantic_similarity", 0)),
            "Sentence-BERT cosine distance",
        )

    table.add_row("", "", "")
    table.add_row(
        "[bold]Composite Score[/bold]",
        f"[bold]{fmt_score(scores['composite_score'])}[/bold]",
        "[bold]Weighted blend (see weights in evaluator.py)[/bold]",
    )

    console.print(table)

    if isinstance(judge, dict) and "overall_comment" in judge:
        console.print(f"  [italic dim]Judge comment: {judge['overall_comment']}[/italic dim]")
    console.print()

    return scores


def custom_mode(generator: EmailResponseGenerator):
    console.print("[bold yellow]Custom email mode[/bold yellow]")
    console.print("Enter the customer email (press Enter twice when done):\n")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    incoming = "\n".join(lines).strip()
    subject = input("Subject (optional, press Enter to skip): ").strip()

    gen = generator.generate(incoming, subject)
    console.print(Panel(gen["generated_reply"], title="Generated Reply", border_style="green"))

    scores = evaluate(incoming, gen["generated_reply"])
    judge = scores.get("llm_judge", {})
    console.print(f"\n[bold]LLM Judge avg:[/bold] {scores['llm_judge_avg']}/10")
    console.print(f"[bold]Composite score:[/bold] {scores['composite_score']}")
    if isinstance(judge, dict) and "overall_comment" in judge:
        console.print(f"[italic]{judge['overall_comment']}[/italic]")


def main():
    parser = argparse.ArgumentParser(description="Hiver Email AI — demo runner")
    parser.add_argument("--dataset", default="data/emails.json")
    parser.add_argument("--custom", action="store_true", help="Enter a custom email interactively")
    parser.add_argument("--id", type=int, help="Evaluate a single email by dataset ID")
    parser.add_argument("--all", action="store_true", help="Evaluate all emails in the dataset")
    parser.add_argument("--top-k", type=int, default=3, help="Retrieved examples per generation")
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "groq"],
        help="LLM provider for generation: 'anthropic' (Claude Sonnet) or 'groq' (Llama 3.3 70B)",
    )
    parser.add_argument("--model", default=None, help="Override the default model for the chosen provider")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    generator = EmailResponseGenerator(args.dataset, top_k=args.top_k, provider=args.provider, model=args.model)

    provider_label = {
        "anthropic": f"Claude Sonnet 4.6 (Anthropic)",
        "groq": f"Llama 3.3 70B (Groq)",
    }.get(args.provider, args.provider)
    if args.model:
        provider_label = f"{args.model} ({args.provider})"

    console.print()
    console.print(Panel.fit(
        "[bold green]Hiver — AI Email Response System[/bold green]\n"
        f"[dim]Generator: {provider_label} + TF-IDF RAG[/dim]\n"
        "[dim]Evaluator: ROUGE + Semantic Similarity + LLM Judge (Claude Haiku)[/dim]",
        border_style="green",
    ))
    console.print()

    if args.custom:
        custom_mode(generator)
        return

    if args.id:
        test_emails = [e for e in dataset if e["id"] == args.id]
        if not test_emails:
            console.print(f"[red]No email with ID {args.id}[/red]")
            sys.exit(1)
    elif args.all:
        test_emails = dataset
    else:
        # Default: last 5 emails as held-out test set
        test_emails = dataset[-5:]

    all_scores = []
    for i, email in enumerate(test_emails, 1):
        scores = run_evaluation(email, generator, i, len(test_emails))
        all_scores.append(scores)

    # ------------------------------------------------------------------ #
    # Overall system summary
    # ------------------------------------------------------------------ #
    console.rule("[bold]Overall System Performance[/bold]")
    summary = Table(box=box.ROUNDED, show_header=True, header_style="bold white")
    summary.add_column("Metric", style="bold")
    summary.add_column("Mean", justify="right")
    summary.add_column("Min", justify="right")
    summary.add_column("Max", justify="right")

    composite = [s["composite_score"] for s in all_scores]
    summary.add_row(
        "Composite Score",
        fmt_score(np.mean(composite)),
        fmt_score(np.min(composite)),
        fmt_score(np.max(composite)),
    )

    llm_avgs = [s.get("llm_judge_avg", 0) for s in all_scores]
    summary.add_row(
        "LLM Judge Avg (out of 10)",
        fmt_score(np.mean(llm_avgs), 10),
        fmt_score(np.min(llm_avgs), 10),
        fmt_score(np.max(llm_avgs), 10),
    )

    rouge_scores_l = [s["rouge"]["rougeL"] for s in all_scores if "rouge" in s]
    if rouge_scores_l:
        summary.add_row(
            "ROUGE-L",
            fmt_score(np.mean(rouge_scores_l)),
            fmt_score(np.min(rouge_scores_l)),
            fmt_score(np.max(rouge_scores_l)),
        )

    sem_sims = [s.get("semantic_similarity", 0) for s in all_scores if "semantic_similarity" in s]
    if sem_sims:
        summary.add_row(
            "Semantic Similarity",
            fmt_score(np.mean(sem_sims)),
            fmt_score(np.min(sem_sims)),
            fmt_score(np.max(sem_sims)),
        )

    console.print(summary)
    console.print(f"\n[dim]Evaluated {len(test_emails)} email(s). Composite = "
                  f"{WEIGHTS_DESC()}[/dim]\n")


def WEIGHTS_DESC():
    return "40% LLM Judge + 35% Semantic Similarity + 25% ROUGE-L"


if __name__ == "__main__":
    main()
