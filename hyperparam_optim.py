import os
import ray
from ray import train, tune
from ray.train import RunConfig, FailureConfig
from ray.tune.schedulers import ASHAScheduler

os.environ["OMP_NUM_THREADS"] = "24"


model = "./models/yolov8n.pt"

model_in_store = ray.put(model)

search_space = {
    # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
    "lr0": tune.uniform(1e-5, 1e-1),
    "lrf": tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
    "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
    "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
    "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
    "box": tune.uniform(0.02, 0.2),  # box loss gain
    "cls": tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
    "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
    "degrees": tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
    "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
    "scale": tune.uniform(0.0, 0.9),  # image scale (+/- gain)
    "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
    "perspective": tune.uniform(
        0.0, 0.001
    ),  # image perspective (+/- fraction), range 0-0.001
    "flipud": tune.uniform(0.0, 1.0),  # image flip up-down (probability)
    "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
    "bgr": tune.uniform(0.0, 1.0),  # image channel BGR (probability)
    "mosaic": tune.uniform(0.0, 1.0),  # image mixup (probability)
    "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
    "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
}


def trainable(search_space):

    model_to_train = ray.get(model_in_store)

    score = "metric to optimize"
    train.report({"score": score})  # Send the score to Tune.


asha_scheduler = ASHAScheduler(
    time_attr="epoch",
    metric=TASK2METRIC[task],
    mode="max",
    max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
    grace_period=5,  # default is 10
    reduction_factor=3,
)

# Define the trainable function with allocated resources
trainable_with_resources = tune.with_resources(
    trainable, resources={"cpu": 6, "gpu": 1}
)

tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        num_samples=4,
        scheduler=asha_scheduler,
    ),
    run_config=RunConfig(
        name="hyperparam_test",
        log_to_file=True,
        failure_config=FailureConfig(fail_fast=True, max_failures=0),
    ),
    param_space=search_space,
)
results = tuner.fit()
