import argparse
import json
import logging
import os
import sys
from typing import Union

from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import (
    handle_non_serializable,
    make_table,
    simple_parse_args_string,
    str_to_dict,
)
from lm_eval.utils import setup_parser, parse_eval_args


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        parser = setup_parser()
        args = parse_eval_args(parser)

    if args.wandb_args:
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # update the evaluation tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError(
            "Specify --output_path if providing --log_samples or --predict_only"
        )

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError(
            "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set (either to `True` or to the chosen template name)."
        )

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning(
            "Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub."
        )

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if not args.is_infinitebench:
        if args.tasks is None:
            eval_logger.error("Need to specify task to evaluate.")
            sys.exit()
        elif args.tasks == "list":
            print(task_manager.list_all_tasks())
            sys.exit()
        elif args.tasks == "list_groups":
            print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
            sys.exit()
        elif args.tasks == "list_tags":
            print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
            sys.exit()
        elif args.tasks == "list_subtasks":
            print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
            sys.exit()
        else:
            if os.path.isdir(args.tasks):
                import glob

                task_names = []
                yaml_path = os.path.join(args.tasks, "*.yaml")
                for yaml_file in glob.glob(yaml_path):
                    config = utils.load_yaml_config(yaml_file)
                    task_names.append(config)
            else:
                task_list = args.tasks.split(",")
                task_names = task_manager.match_tasks(task_list)
                for task in [task for task in task_list if task not in task_names]:
                    if os.path.isfile(task):
                        config = utils.load_yaml_config(task)
                        task_names.append(config)
                task_missing = [
                    task
                    for task in task_list
                    if task not in task_names and "*" not in task
                ]  # we don't want errors if a wildcard ("*") task name was used

                if task_missing:
                    missing = ", ".join(task_missing)
                    eval_logger.error(
                        f"Tasks were not found: {missing}\n"
                        f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                    )
                    raise ValueError(
                        f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                    )
    else:
        task_names = args.tasks
    eval_logger.info(f"Selected Tasks: {task_names}")

    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if args.trust_remote_code:
        eval_logger.info(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        args.model_args = args.model_args + ",trust_remote_code=True"

    request_caching_args = request_caching_arg_to_dict(
        cache_requests=args.cache_requests
    )

    # [x] Process Tiered KVCache Arguments
    if isinstance(args.model_args, str):
        model_args = str_to_dict(args.model_args)
    if args.cache_args != None:
        cache_args = str_to_dict(args.cache_args)
        assert "algo" in cache_args
        assert "cache_rule" in cache_args

        if cache_args["algo"] in {"ideal"}:
            assert "cache_ratio" in cache_args
        if cache_args["algo"] in {"thresholding"}:
            assert "alpha" in cache_args

        if "alpha" in cache_args:
            cache_args["alpha"] = float(cache_args["alpha"])
        if "cache_ratio" in cache_args:
            cache_args["cache_ratio"] = float(cache_args["cache_ratio"])
        if "decay_rate" in cache_args:
            cache_args["decay_rate"] = float(cache_args["decay_rate"])

        for key in cache_args:
            model_args[key] = cache_args[key]
    ###

    print(model_args)

    if args.is_infinitebench:
        evaluator.infinitebench_evaluate(
            model=args.model,
            model_args=model_args,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            use_cache=args.use_cache,
            limit=args.limit,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            log_samples=args.log_samples,
            evaluation_tracker=evaluation_tracker,
            system_instruction=args.system_instruction,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            gen_kwargs=args.gen_kwargs,
            task_manager=task_manager,
            verbosity=args.verbosity,
            predict_only=args.predict_only,
            random_seed=args.seed[0],
            numpy_random_seed=args.seed[1],
            torch_random_seed=args.seed[2],
            fewshot_random_seed=args.seed[3],
            data_dir=args.data_dir,
            max_seq_length=args.max_seq_length,
            output_dir=args.output_dir,
            num_eval_examples=args.num_eval_examples,
            rewrite=args.rewrite,
            start_example_id=args.start_example_id,
            **request_caching_args,
        )
    else:
        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=model_args,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            use_cache=args.use_cache,
            limit=args.limit,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            log_samples=args.log_samples,
            evaluation_tracker=evaluation_tracker,
            system_instruction=args.system_instruction,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            gen_kwargs=args.gen_kwargs,
            task_manager=task_manager,
            verbosity=args.verbosity,
            predict_only=args.predict_only,
            random_seed=args.seed[0],
            numpy_random_seed=args.seed[1],
            torch_random_seed=args.seed[2],
            fewshot_random_seed=args.seed[3],
            **request_caching_args,
        )

        if results is not None:
            if args.log_samples:
                samples = results.pop("samples")
            dumped = json.dumps(
                results, indent=2, default=handle_non_serializable, ensure_ascii=False
            )
            if args.show_config:
                print(dumped)

            batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

            # Add W&B logging
            if args.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if args.log_samples:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

            evaluation_tracker.save_results_aggregated(
                results=results, samples=samples if args.log_samples else None
            )

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    evaluation_tracker.save_results_samples(
                        task_name=task_name, samples=samples[task_name]
                    )

            if (
                evaluation_tracker.push_results_to_hub
                or evaluation_tracker.push_samples_to_hub
            ):
                evaluation_tracker.recreate_metadata_card()

            print(
                f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
                f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
            )
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

            if args.wandb_args:
                # Tear down wandb run once all the logging is done.
                wandb_logger.run.finish()


if __name__ == "__main__":
    cli_evaluate()
