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
        ("train", "Train the full 450/50 pilot"),
        ("overfit-smoke", "Run the fixed-example overfit smoke test"),
        ("evaluate", "Evaluate a saved adapter on validation"),
        ("reload-test", "Verify deterministic adapter reload generation"),
    ):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--config", default=str(DEFAULT_CONFIG))
        item.add_argument("--cache-dir", required=True)
        item.add_argument("--output-dir", required=True)
        if name in {"evaluate", "reload-test"}:
            item.add_argument("--adapter", required=True)
        if name == "overfit-smoke":
            item.add_argument("--examples", type=int)
            item.add_argument("--max-steps", type=int)
            item.add_argument("--required-loss-ratio", type=float)
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
    else:
        from .trainer import evaluate, overfit_smoke, reload_test, train_full
        if args.command == "train":
            train_full(args.config, args.cache_dir, args.output_dir)
        elif args.command == "overfit-smoke":
            overfit_smoke(args.config, args.cache_dir, args.output_dir,
                          args.examples, args.max_steps, args.required_loss_ratio)
        elif args.command == "evaluate":
            evaluate(args.config, args.cache_dir, args.output_dir, args.adapter)
        elif args.command == "reload-test":
            reload_test(args.config, args.cache_dir, args.output_dir, args.adapter)


if __name__ == "__main__":
    main()
