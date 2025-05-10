import gymnasium as gym
import vizdoom.gymnasium_wrapper  # noqa: F401

import pretrained.envs  # noqa: F401
from src.data.common import (
    DEFAULT_REWARD_FUNCS,
    DEFEND_CENTER_REWARD_FUNCS,
    HEALTH_GATHERING_REWARD_FUNCS,
)
from src.data.wrappers import VizdoomWithRewardWrapper

# from src.data.common import DOOM_BUTTONS

# from src.data.wrappers import VizdoomEditGameWrapper
# from pretrained.envs import VizdoomArnoldWithLabelsBuffer

gym.register(
    "sa/DefendLine-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_line.cfg"},
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFAULT_REWARD_FUNCS),
    ),
)

gym.register(
    "sa/DefendCenter-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "defend_the_center.cfg"},
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFEND_CENTER_REWARD_FUNCS),
    ),
)

gym.register(
    "sa/Deathmatch-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "deathmatch.cfg"},
    additional_wrappers=(
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFAULT_REWARD_FUNCS),
    ),
)

_arnold_dtc_spec = gym.spec("arnold/DefendCenter-v0")
gym.register(
    "sa/ArnoldDefendCenter-v0",
    _arnold_dtc_spec.entry_point,
    kwargs=_arnold_dtc_spec.kwargs,
    additional_wrappers=(
        *_arnold_dtc_spec.additional_wrappers,
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFEND_CENTER_REWARD_FUNCS),
    ),
)
_arnold_dtc_spec = gym.spec("arnold/HealthGathering-v0")
gym.register(
    "sa/ArnoldHealthGathering-v0",
    _arnold_dtc_spec.entry_point,
    kwargs=_arnold_dtc_spec.kwargs,
    additional_wrappers=(
        *_arnold_dtc_spec.additional_wrappers,
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(
            reward_funcs=HEALTH_GATHERING_REWARD_FUNCS
        ),
    ),
)


# _arnold_dtc_spec = gym.spec("sa/Deathmatch-v0")
# gym.register(
#     "sa/ArnoldDeathmatch-v0",
#     _arnold_dtc_spec.entry_point,
#     kwargs=_arnold_dtc_spec.kwargs,
#     additional_wrappers=(
#         gym.wrappers.AutoResetWrapper.wrapper_spec(),
#         VizdoomEditGameWrapper.wrapper_spec(
#             edit_fn=partial(pretrained.envs.vizdoom_to_arnold_game_edit_fn, binary_buttons=DOOM_BUTTONS, set_render_options=True)
#         ),
#         VizdoomArnoldWithLabelsBuffer.wrapper_spec(
#             labels_mapping="0",
#             width=108,
#             height=60,
#         ),
#         VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFAULT_REWARD_FUNCS),
#     ),
# )

_arnold_dtc_spec = gym.spec("arnold/Shotgun-v0")
gym.register(
    "sa/ArnoldShotgun-v0",
    _arnold_dtc_spec.entry_point,
    kwargs=_arnold_dtc_spec.kwargs,
    additional_wrappers=(
        *_arnold_dtc_spec.additional_wrappers,
        gym.wrappers.AutoResetWrapper.wrapper_spec(),
        VizdoomWithRewardWrapper.wrapper_spec(reward_funcs=DEFAULT_REWARD_FUNCS),
    ),
)
