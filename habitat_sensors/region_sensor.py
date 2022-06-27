from typing import Any
import numpy as np
from gym import spaces
import habitat

HABITAT_ROOMS = ['unknown', 'familyroom/lounge', 'laundryroom/mudroom', 'kitchen', 'entryway/foyer/lobby', 'bedroom', 'bathroom', 'living room', 'stairs', 'hallway', 'balcony', 'lounge', 'office', 'closet', 'toilet', 'porch/terrace/deck', 'dining room', 'garage', 'other room', 'workout/gym/exercise', 'spa/sauna', 'junk', 'utilityroom/toolroom', 'rec/game', 'outdoor', 'tv', 'meetingroom/conferenceroom', 'bar', 'library']

@habitat.registry.register_sensor(name="current_region_name")
class CurrentRegionNameSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim
        self.scene_rooms = {}

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "current_region_name"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.TEXT

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Discrete(18)
        #return spaces.Discrete(len(HABITAT_ROOMS))

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        room_name = "unknown"

        semantic_annotations = self._sim.semantic_annotations()

        pt = self._sim.get_agent_state().position

        if semantic_annotations is not None:
            for region in semantic_annotations.regions:
                result = np.all(((region.aabb.center - abs(region.aabb.sizes) / 2) <= pt) & ((region.aabb.center + abs(region.aabb.sizes) / 2) >= pt))
                if region.category is None:
                    return room_name
                room_name = region.category.name()
                if result:
                    return room_name

        # nothing found
        return room_name
