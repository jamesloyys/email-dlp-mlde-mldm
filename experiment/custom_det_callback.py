import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import determined as det
import jsonlines
import torch

logger = logging.getLogger("determined.transformers")


class DetCallback(TrainerCallback):  # type: ignore
    def __init__(
        self,
        core_context: det.core.Context,
        args: TrainingArguments,
        filter_metrics: Optional[List[str]] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.core_context = core_context

        self.filter_metrics = filter_metrics
        self.user_data = user_data
        self.load_last_checkpoint(args)

        self.last_metrics: Dict[str, float] = {"train_step": -1, "eval_step": -1}
        self.searcher_ops = self.core_context.searcher.operations()
        self.current_op = next(self.searcher_ops)
        self.updating_searcher = False

        cluster_info = det.get_cluster_info()
        assert (
            cluster_info
        ), "Could not find `cluster_info`, the HF Callback must be run on a Determined Cluster"
        searcher_config = cluster_info.trial._config["searcher"]
        self.searcher_metric = searcher_config["metric"]
        # Custom searchers have a different config structure which need to be handled differently
        if searcher_config["name"] == "custom":
            self.searcher_unit = "batches"
            self.searcher_max_length = self.current_op.length
        else:
            self.searcher_unit = list(searcher_config["max_length"].keys())[0]
            self.searcher_max_length = list(searcher_config["max_length"].values())[0]
            self._check_searcher_compatibility(args)

    def set_model(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def set_test_data(self, test_data_path, test_data_ground_truth_path):
        self.test_data_path = test_data_path
        self.test_data_ground_truth_path = test_data_ground_truth_path

    def _calculate_metrics(self, actual_categories, predicted_categories):
        true_positive, false_negative, false_positive = 0, 0, 0
        if actual_categories == [] and predicted_categories == []:
            true_positive += 1
        elif actual_categories == [] and predicted_categories != []:
            false_negative += 1
        else:
            for c in actual_categories:
                if c in predicted_categories:
                    true_positive += 1
            false_negative = len(actual_categories) - true_positive
        false_positive = max(len(predicted_categories) - true_positive, 0)
        return true_positive, false_negative, false_positive

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.model.eval()
        logger.warning(f"Model is: {type(self.model)}")

        with jsonlines.open(self.test_data_path) as reader:
            emails = [email for email in reader]
        with open(self.test_data_ground_truth_path, "rb") as f:
            eval_labels = json.load(f)

        all_metrics = {}

        for idx in range(0, len(emails)):
            email = emails[idx]
            prompt = email['text'][:email['text'].find('[/INST]')+7]
            actual_categories = eval_labels[str(idx)]["restricted information found in email"]
            actual_categories = [k for k,v in actual_categories.items() if v != []]
            logger.warning(f"Index {idx} | Actual categories is: {actual_categories}")
            
            model_input = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            with torch.autocast(device_type='cuda'):
                output = self.tokenizer.decode(self.model.generate(**model_input, temperature=0.01, top_p=0.9, do_sample=True, max_new_tokens=500)[0], skip_special_tokens=True)
            logger.warning(output)
            try:
                predicted_categories = output[output.find('[/INST]')+7:output.find('</s>')]
                predicted_categories = json.loads(predicted_categories, strict=False)['restricted information found in email']
                predicted_categories = [k for k,v in predicted_categories.items() if v != []]
            except:
                predicted_categories = output[output.find('[/INST]')+7:output.find('</s>')]
                print(predicted_categories)
                last_comma_idx = predicted_categories.rfind(",")
                predicted_categories = predicted_categories[:last_comma_idx] + predicted_categories[last_comma_idx+1:]
                predicted_categories = [k for k,v in predicted_categories.items() if v != []]

            logger.warning(f"Index {idx} | Predicted categories is: {predicted_categories}")

            true_positive, false_negative, false_positive = self._calculate_metrics(actual_categories, predicted_categories)
            all_metrics[idx] = {"true positive": true_positive, "false negative": false_negative, "false positive": false_positive}

        all_tp, all_fn, all_fp = 0,0,0
        for idx in all_metrics:
            all_tp += all_metrics[idx]["true positive"]
            all_fn += all_metrics[idx]["false negative"]
            all_fp += all_metrics[idx]["false positive"]
        all_recall = all_tp/(all_tp+all_fn)
        logger.warning(f"Overall recall is {all_recall}")

        self.core_context.train.report_validation_metrics(
            steps_completed=state.global_step, metrics={"train_acc": all_recall}
        )
        

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            logger.warning("on_log called with empty logs")
            return
        metrics, metric_type = self._get_metrics(logs)
        logger.debug(f"on_log metrics, global_step {state.global_step}", metrics)
        if metric_type == TRAIN:
            # Prevents reporting metrics for the same step twice. This happens after
            # training is completed and average training metrics are reported with
            # the same step as the in-progress training metrics.
            if self.last_metrics["train_step"] != state.global_step:
                if state.is_world_process_zero:
                    self.core_context.train.report_training_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                metrics["train_step"] = state.global_step

        elif metric_type == EVAL:
            # Prevents reporting metrics for the same step twice. This happens when
            # after-training evaluation is completed, and it is reported with the same
            # step as the last during-training evaluation.
            if self.last_metrics["eval_step"] != state.global_step:
                if state.is_world_process_zero:
                    self.core_context.train.report_validation_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                metrics["eval_step"] = state.global_step
        else:
            logger.warning(f"Metrics not reported: metric type = {metric_type}.")

        self.last_metrics.update(metrics)

        # Update searcher state after collecting the metrics.
        if self.updating_searcher is True:
            self._update_searcher(state, control)

        # If searcher is NOT being updated and preemption signal is received
        # (e.g., by pausing experiment in the WebUI), notify Trainer (via TrainerControl)
        # to save the checkpoint. After the checkpoint is uploaded to Determined storage,
        # the process is preempted (see on_save() method for details).
        if self.updating_searcher is False and self.core_context.preempt.should_preempt():
            control.should_save = True

    def _get_metrics(self, logs: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        metrics = logs
        metric_type = get_metric_type(logs)
        if self.filter_metrics:
            metrics = {}
            for k, v in logs.items():
                if any(m in k for m in self.filter_metrics) is True:
                    metrics[k] = v

        return metrics, metric_type

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        info = det.get_cluster_info()
        assert info

        # local_path is where HF Trainer saves model and tokenizer in a given step.
        local_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if state.is_world_process_zero:
            if self.user_data is not None:
                self._on_save_user_data(local_path)

        det_checkpoint_metadata = {
            "steps_completed": state.global_step,
            "trial_id": info.trial.trial_id,
        }

        def selector(x: str) -> bool:
            return x.startswith((f"checkpoint-{state.global_step}/", "runs/"))

        self.core_context.checkpoint.upload(
            args.output_dir, metadata=det_checkpoint_metadata, shard=True, selector=selector
        )

        if self.core_context.preempt.should_preempt():
            raise Exception("Process preempted / killed")

    def _on_save_user_data(self, save_path: str) -> None:
        """
        User-defined saving of objects from self.checkpoint_metadata under save_path.
        After objects are saved, Determined handles uploading and downloading objects
        to/from selected storage.
        """
        with open(os.path.join(save_path, "my_data.json"), "w") as f:
            json.dump(self.user_data, f)

    def load_last_checkpoint(self, args: TrainingArguments) -> None:
        info = det.get_cluster_info()
        assert info

        latest_checkpoint = info.latest_checkpoint
        if latest_checkpoint is not None:
            if args.overwrite_output_dir is True:
                logger.info(
                    "Skip downloading last checkpoint from Determined due "
                    "to overwrite_output_dir=True."
                )
                return

            # To resume DeepSpeed, each node requires ALL sharded model/optimizer states,
            # so we can skip using selector and just download all files.
            self.core_context.checkpoint.download(latest_checkpoint, args.output_dir)

            checkpoint_path = get_last_checkpoint(args.output_dir)
            args.resume_from_checkpoint = checkpoint_path

            logger.info(f"Latest checkpoint downloaded to {checkpoint_path}.")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # state.epoch is not None only during training.
        if state.epoch and self.searcher_unit == "batches":
            if state.is_world_process_zero:
                self.current_op.report_progress(state.global_step)

            if state.global_step >= self.current_op.length:
                logger.info(
                    f"Max length of {self.current_op.length} steps reached for current "
                    f"searcher operation. Updating searcher."
                )
                self._update_searcher(state, control)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # state.epoch is not None only during training.
        if state.epoch and self.searcher_unit == "epochs":
            if state.is_world_process_zero:
                self.current_op.report_progress(state.epoch)

            if state.epoch >= self.current_op.length:
                logger.info(
                    f"Max length of {state.epoch} epochs reached for current "
                    f"searcher operation. Updating searcher."
                )
                self._update_searcher(state, control)

    def _update_searcher(self, state: TrainerState, control: TrainerControl) -> None:
        if self._metrics_reported(state.global_step) is False:
            self._wait_for_metrics(control)
            return

        if state.is_world_process_zero:
            if self.last_metrics is None:
                logger.warning(
                    "No training or evaluation metrics has been recorded. Please "
                    "check your settings for training metrics "
                    "(--logging_strategy and --logging_steps) or "
                    "evaluation metrics (--evaluation_strategy and --eval_steps). "
                    "Reporting trainer_state.best_metric to the searcher."
                )
                searcher_metric = state.best_metric
            elif self.searcher_metric not in self.last_metrics:
                logger.warning(
                    f"Searcher metric {self.searcher_metric} from the yaml config file does "
                    "not match any of the recorded metrics "
                    f"in {self.last_metrics}. "
                    "Reporting trainer_state.best_metric to the searcher."
                )
                searcher_metric = state.best_metric
            else:
                searcher_metric = self.last_metrics[self.searcher_metric]

            logger.info(f"Metric reported to searcher: {searcher_metric}")
            self.current_op.report_completed(searcher_metric)

        self.updating_searcher = False

        try:
            self.current_op = next(self.searcher_ops)
        except StopIteration:
            control.should_training_stop = True

    def _metrics_reported(self, step: int) -> bool:
        return self.last_metrics["eval_step"] == step and self.last_metrics["train_step"] == step

    def _wait_for_metrics(self, control: TrainerControl) -> None:
        # Notify Trainer (via TrainerControl) to:
        # (1) log current training metrics,
        # (2) evaluate the model and log evaluation metrics,
        # (3) save the checkpoint.
        #  updating_searcher is as an internal flag that indicates we are
        #  in the process of updating the searcher with the current metrics.
        control.should_log = True
        control.should_evaluate = True
        control.should_save = True
        self.updating_searcher = True

    def _check_searcher_compatibility(self, args: TrainingArguments) -> None:
        if self.searcher_unit == "batches":
            if args.max_steps == -1:
                self._log_config_mismatch("epochs", args.num_train_epochs)
            elif args.max_steps != self.searcher_max_length:
                self._log_config_mismatch("batches", args.max_steps)
        elif self.searcher_unit == "epochs":
            if args.max_steps != -1:
                self._log_config_mismatch("batches", args.max_steps)
            elif args.num_train_epochs != self.searcher_max_length:
                self._log_config_mismatch("epochs", args.num_train_epochs)

    def _log_config_mismatch(
        self,
        trainer_units: str,
        trainer_len: float,
    ) -> None:
        logger.warning(
            f"Searcher configuration does not match HF Trainer configuration. "
            f"Searcher uses {self.searcher_unit}={self.searcher_max_length}, "
            f"while HF Trainer uses {trainer_units}={trainer_len}. "
            f"Continuing this run may cause Searcher not to behave correctly. "
            f"Make sure to match the units between HF Trainer and Searcher: "
            f"use (--num_train_epochs and searcher.max_length.epochs) OR "
            f"(--max_steps and searcher.max_length.batches)."
        )


EVAL = "eval_"
TEST = "test_"
TRAIN_AVG = "train_"
TRAIN = "train_progress"


def get_metric_type(d: Dict[str, Any]) -> str:
    for k, _ in d.items():
        if k.startswith(EVAL):
            return EVAL
        elif k.startswith(TEST):
            return TEST
        elif k.startswith(TRAIN_AVG):
            return TRAIN_AVG
        else:
            return TRAIN
    return TRAIN


def get_ds_config_path_from_args(args: List[str]) -> Optional[str]:
    for idx in range(len(args)):
        if args[idx] == "--deepspeed":
            ds_config_idx = idx + 1
            ds_config_path = args[ds_config_idx]
            return ds_config_path
    return None
