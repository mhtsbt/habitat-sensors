from typing import Any
import numpy as np
from gym import spaces
import habitat


@habitat.registry.register_sensor(name="goal_mask")
class GoalMaskSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "goal_mask"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Discrete(18)

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        semantic_ids = list(map(lambda x: x.semantic_id, (filter(lambda x: x.category.name() == episode.object_category, self._sim.semantic_scene.objects))))

        mask = list(map(lambda x: x in semantic_ids, np.reshape(observations['semantic'], -1)))
        mask = np.reshape(mask, (480, 640))

        return mask
