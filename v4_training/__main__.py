"""Command-line entry point for V4 audits and LoRA training."""

import argparse

from .config import DEFAULT_CONFIG


def parser():
    command = argparse.ArgumentParser(prog="python -m v4_training")
    sub = command.add_subparsers(dest="command", required=True)

    def audit_args(name, help_text):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--config", default=str(DEFAULT_CONFIG))
        item.add_argument("--cache-dir")
        item.add_argument("--local-files-only", action="store_true")
        item.add_argument("--examples", type=int, default=3)
        return item

    tokenizer = audit_args("audit-tokenizer", "Audit the pinned tokenizer and chat template")
    tokenizer.add_argument("--report")
    audit_args("audit-labels", "Audit completion-only labels")
    modules = audit_args("audit-modules", "Audit LoRA targets from the pinned weight index")
    modules.add_argument("--report")

    for name, help_text in (
        ("train", "Train the full configured dataset"),
        ("overfit-smoke", "Run the fixed-example overfit smoke test"),
        ("evaluate", "Evaluate a saved adapter on validation"),
        ("evaluate-cases", "Record completion-only loss for every validation case"),
        ("reload-test", "Verify deterministic adapter reload generation"),
    ):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--config", default=str(DEFAULT_CONFIG))
        item.add_argument("--cache-dir", required=True)
        item.add_argument("--output-dir", required=True)
        if name in {"evaluate", "evaluate-cases", "reload-test"}:
            item.add_argument("--adapter", required=True)
        if name == "overfit-smoke":
            item.add_argument("--examples", type=int)
            item.add_argument("--max-steps", type=int)
            item.add_argument("--required-loss-ratio", type=float)

    comparison = sub.add_parser(
        "compare-adapters", help="Generate a fixed side-by-side adapter sample")
    comparison.add_argument("--config", default=str(DEFAULT_CONFIG))
    comparison.add_argument("--cache-dir", required=True)
    comparison.add_argument("--output-dir", required=True)
    comparison.add_argument("--adapter-a", required=True)
    comparison.add_argument("--adapter-b", required=True)
    comparison.add_argument("--examples", type=int, default=8)
    comparison.add_argument("--max-new-tokens", type=int, default=768)

    merge = sub.add_parser(
        "merge-adapter", help="Merge and verify a LoRA adapter into the pinned base")
    merge.add_argument("--config", default=str(DEFAULT_CONFIG))
    merge.add_argument("--cache-dir", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--verification-tokens", type=int, default=128)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "audit-tokenizer":
        from .audit import audit_tokenizer
        audit_tokenizer(args.config, args.cache_dir, args.local_files_only, args.examples, args.report)
    elif args.command == "audit-labels":
        from .audit import audit_labels
        audit_labels(args.config, args.cache_dir, args.local_files_only, args.examples)
    elif args.command == "audit-modules":
        from .audit import audit_weight_index
        audit_weight_index(args.config, args.cache_dir, args.local_files_only, args.report)
    elif args.command == "compare-adapters":
        from .trainer import compare_adapters
        compare_adapters(
            args.config, args.cache_dir, args.output_dir, args.adapter_a,
            args.adapter_b, args.examples, args.max_new_tokens)
    elif args.command == "merge-adapter":
        from .trainer import merge_adapter
        merge_adapter(
            args.config, args.cache_dir, args.output_dir, args.adapter,
            args.verification_tokens)
    else:
        from .trainer import evaluate, evaluate_cases, overfit_smoke, reload_test, train_full
        if args.command == "train":
            train_full(args.config, args.cache_dir, args.output_dir)
        elif args.command == "overfit-smoke":
            overfit_smoke(args.config, args.cache_dir, args.output_dir,
                          args.examples, args.max_steps, args.required_loss_ratio)
        elif args.command == "evaluate":
            evaluate(args.config, args.cache_dir, args.output_dir, args.adapter)
        elif args.command == "evaluate-cases":
            evaluate_cases(args.config, args.cache_dir, args.output_dir, args.adapter)
        elif args.command == "reload-test":
            reload_test(args.config, args.cache_dir, args.output_dir, args.adapter)


if __name__ == "__main__":
    main()
