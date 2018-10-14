from math import *
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

def is_collision(entity1, entity2):
    """Determine if two entities have collided."""
    a, b     = entity1.state.p_pos, entity2.state.p_pos
    dist     = np.linalg.norm(a - b)
    min_dist = entity1.size + entity2.size
    return dist < min_dist**2

class HelloMyNameIs(Agent):
    """Instance of this class have a name that every agent can 'see'."""
    def __init__(self, silent=True):
        super(HelloMyNameIs, self).__init__()
        self.identity = np.random.randint(0, 100)
        self.reputation = np.zeros((100))
        self.reputation[self.identity] = 1
        self.true_color = None
        self.carrying_food = False
        self.stole_food = False

        self.color = np.array([.5, .9, .5])
        self.collide = True
        self.silent = silent
        self.size = 0.05
        self.accel = 1.0
        self.max_speed = 1.0
        self.name = "pleb %d" % self.identity

    def reward(self, world):
        # All agents stay in-bounds
        def bound1(x):
            if x < .9:
                return 0
            elif x < 1.0:
                return (x - .9)*10
            else:
                return min(exp(2*x - 2), 10)

        reward = 0
        for p in range(world.dim_p):
            x = abs(self.state.p_pos[p])
            reward -= bound1(x)
        return reward

class Pleb(HelloMyNameIs):
    def __init__(self, food_source, nest):
        super(Pleb, self).__init__()
        self.true_color = np.array([.5, .9, .5])

        self.food_source = food_source
        self.nest = nest

        self.objectives = [nest, food_source]
        self.objective_rewards = [self._dropoff_food, self._pickup_food]
        self.current_objective = 1

    def reward(self, world, scenario):
        reward = super(Pleb, self).reward(world)
        assert(not is_collision(self, self.nest)
               or not is_collision(self, self.food_source))

        # Get a reward if we have achieved our current objective.
        obj = self.current_objective
        spatial_objective = self.objectives[obj]
        if is_collision(self, spatial_objective):
            reward += self.objective_rewards[obj]()
            spatial_objective = self._next_objective()

        # Reward is inversely proportional to distance to spatial_objective
        (o, s) = (spatial_objective.state.p_pos, self.state.p_pos)
        reward += np.reciprocal(np.sqrt(np.linalg.norm(o - s)))
        return reward

    def _dropoff_food(self):
        self.carrying_food = False
        return 10

    def _pickup_food(self):
        self.carrying_food = True
        return 10

    def _next_objective(self):
        obj = self.current_objective
        self.current_objective = obj = (1 + obj) % len(self.objectives)
        return self.objectives[obj]

class Spy(HelloMyNameIs):
    def __init__(self):
        super(Spy, self).__init__()
        self.true_color = np.array([.9, .5, .5])

    def reward(self, world):
        reward = super(Spy, self).reward(world)
        return reward

class Scenario(BaseScenario):
    def make_world(self, use_reputation=True):
        self.world = world = World()
        self.use_reputation = use_reputation

        world.dim_c = 2

        self.food_source = food_source = Landmark()
        food_source.name = 'food-source'
        food_source.collide = True
        food_source.movable = False
        food_source.size = 0.2
        food_source.boundary = False
        food_source.color = np.array([.3, .9, .3])

        self.nest = nest = Landmark()
        nest.name = 'nest'
        nest.collide = True
        nest.movable = False
        nest.size = 0.2
        nest.boundary = False
        nest.color = np.array([.9, .7, .5])

        world.landmarks = [food_source, nest]

        # Add agents.
        world.agents = \
            [Pleb(food_source, nest) for i in range(5)]
        #-    [Spy()  for i in range(3)]

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # Set random initial states for all agents.
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # Set random positions for landmarks.
        for landmark in world.landmarks:
            #-if landmark.boundary: continue
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def observation(self, agent, world):
        # Get positions of all landmarks in this agent's reference frame.
        landmark_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                landmark_pos.append(entity.state.p_pos - agent.state.p_pos)

        # Every agent knows where every other agent is and what direction
        # they are headed in.  Use coordinates relative to the agent so
        # that the policies can be generalized.
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmark_pos + other_pos + other_vel)

    def reward(self, agent, world):
        return agent.reward(world, self)

    def benchmark_data(self, agent, world):
        pass
