import dataclasses
import gc
import datetime as dt
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import requests
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model2libero_interface import ModelClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 50  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    log_path: str = "experiments/libero/logs"

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"
    action_chunk_size_override: int | None = None
    summary_json_path: str | None = None
    global_success_offset: int = 0
    global_episode_offset: int = 0


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        action_chunk_size_override=args.action_chunk_size_override,
    )

    disturb_res = {}
    LIBERO_HOME = os.environ.get('LIBERO_HOME', 'path_to_LIBERO-plus_home')
    with open(os.path.join(LIBERO_HOME,'libero/libero/benchmark/task_classification.json')) as f:
        TASK_MAPPING = json.load(f)[args.task_suite_name]
    ID2CATEGORY = {}
    for item in TASK_MAPPING:
        category = item["category"]
        item_name = item["name"]
        ID2CATEGORY[item['id']] = (category, item_name)
        if category not in disturb_res:
            disturb_res[category] = {"total_count": 0, "success_count": 0, "task_count": 0}
        disturb_res[category]["task_count"] += 1

    # Start evaluation

    total_episodes, total_successes = 0, 0
    task_results = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        
        # Get task
        task = task_suite.get_task(task_id)
        task_category, task_name = ID2CATEGORY.get(
            task_id + 1,
            ID2CATEGORY.get(task_id, ("unknown", task.language.replace(" ", "_"))),
        )

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0
            
            # full_actions = np.load("./debug/action.npy")
            
            while t < max_steps + args.num_steps_wait:
                
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = { # 
                    "observation.primary": np.expand_dims(
                        img, axis=0
                    ),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> 这里给了两个图像 --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }
              
                start_time = time.time()
                
                # response = client_model.step(example=example_dict) 
                response = client_model.step(example=example_dict, step=step) 
                
                end_time = time.time()
                # print(f"time: {end_time - start_time}")
                
                # # 
                raw_action = response["raw_action"]
                
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(f"Unexpected action sizes: "
                                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                    f"Falling back to LIBERO_DUMMY_ACTION.")
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)
                
                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    disturb_res.setdefault(task_category, {"total_count": 0, "success_count": 0, "task_count": 0})
                    disturb_res[task_category]["success_count"] += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            disturb_res.setdefault(task_category, {"total_count": 0, "success_count": 0, "task_count": 0})
            disturb_res[task_category]["total_count"] += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")

            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_name}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=25,
            )
            
            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)
            
            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results for this task.
        task_rate = float(task_successes) / float(task_episodes) if task_episodes else 0.0
        suite_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
        global_successes = args.global_success_offset + total_successes
        global_episodes = args.global_episode_offset + total_episodes
        global_rate = float(global_successes) / float(global_episodes) if global_episodes else 0.0
        task_results.append({
            "task_id": int(task_id),
            "task_description": task_description,
            "successes": int(task_successes),
            "episodes": int(task_episodes),
            "success_rate": task_rate,
            "suite_successes_so_far": int(total_successes),
            "suite_episodes_so_far": int(total_episodes),
            "suite_success_rate_so_far": suite_rate,
            "global_successes_so_far": int(global_successes),
            "global_episodes_so_far": int(global_episodes),
            "global_success_rate_so_far": global_rate,
        })
        logging.info(f"Finished task: {task_description}")
        logging.info(f"Current task success rate: {task_rate:.4f} ({task_successes}/{task_episodes})")
        logging.info(f"Current suite success rate: {suite_rate:.4f} ({total_successes}/{total_episodes})")
        logging.info(f"Current global success rate: {global_rate:.4f} ({global_successes}/{global_episodes})")
        try:
            env.close()
        except Exception as exc:
            logging.warning(f"Failed to close LIBERO env cleanly: {exc}")
        del env
        gc.collect()

    pathlib.Path(args.log_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.log_path, f'{args.task_suite_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(disturb_res, f, indent=2)

    final_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    logging.info(f"Total success rate: {final_rate}")
    logging.info(f"Total episodes: {total_episodes}")

    if args.summary_json_path:
        summary = {
            "task_suite_name": args.task_suite_name,
            "total_successes": int(total_successes),
            "total_episodes": int(total_episodes),
            "success_rate": final_rate,
            "global_success_offset": int(args.global_success_offset),
            "global_episode_offset": int(args.global_episode_offset),
            "global_total_successes": int(args.global_success_offset + total_successes),
            "global_total_episodes": int(args.global_episode_offset + total_episodes),
            "global_success_rate": (
                float(args.global_success_offset + total_successes)
                / float(args.global_episode_offset + total_episodes)
                if args.global_episode_offset + total_episodes else 0.0
            ),
            "task_results": task_results,
            "category_results": disturb_res,
        }
        summary_path = pathlib.Path(args.summary_json_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        logging.info(f"Summary JSON saved to: {summary_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    # Only an explicit DEBUG=1/true/yes/on starts debugpy (DEBUG=0 or an empty value must not block).
    if os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        start_debugpy_once()
    tyro.cli(eval_libero)
