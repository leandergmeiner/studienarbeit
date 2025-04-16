from tensordict import TensorDict
import torchrl


class SaveOriginalValuesTransform(torchrl.envs.Transform):
    def __init__(self, in_keys=["observation"], out_key="original"):
        super().__init__(
            in_keys=in_keys, out_keys=[(out_key, in_key) for in_key in in_keys]
        )

    # Simple identitiy-operations here
    def _apply_transform(self, tensordict: TensorDict):
        return tensordict.clone()

    def _inv_apply_transform(self, tensordict: TensorDict):
        # We don't clone here, since we don't care if the saved tensors
        # are modified after the inverse operation
        return tensordict
