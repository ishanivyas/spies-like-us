from math import *
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

GREY     = np.array([.5, .5, .5])
GRAY     = GREY
PURPLE   = np.array([.5, .1, .5])
GREEN    = np.array([.5, .9, .5])
CYAN     = np.array([.0, .9, .9])
BROWN    = np.array([.9, .7, .5])
FAR_AWAY = 10000.0

REWARD_PICKUP_FOOD = 7
REWARD_DROPOFF_FOOD = 13
REWARD_STEAL_FOOD = 1.1*(REWARD_PICKUP_FOOD + REWARD_DROPOFF_FOOD)

def is_collision(entity1, entity2):
    """Determine if two entities have collided."""
    a, b     = entity1.state.p_pos, entity2.state.p_pos
    dist     = np.sqrt(np.sum(np.square(a - b)))
    min_dist = entity1.size + entity2.size
    return dist < min_dist

def spawn_entity(entity, world, pos=None):
    """Place an entity in the world at a random position with 0 velocity."""
    entity.state.p_pos = pos if pos is not None else np.random.uniform(-.75, +.75, world.dim_p)
    entity.state.p_vel = np.zeros(world.dim_p)
    if hasattr(entity.state, "c"):
        entity.state.c = np.zeros(world.dim_c)

def reward_closer(agent, entity, min_dist):
    """Reward an agent for getting closer to an entity."""
    # Agent reward is inversely proportional to distance to entity.  Note
    # however the agent can procrastinate in order to win easy rewards quickly.
    # The min_dist helps eliminate procrastination by tightening the distance
    # that leads to rewards strictly monotonically.  The -.0001 below is meant
    # to account for inconsistencies in floating point comparison.
    (a, e) = (agent.state.p_pos, entity.state.p_pos)
    dist = np.sqrt(np.sum(np.square(a - e)))
    if False:  # Set this to True to avoid using the monotonically tightening reward function.
        return (min_dist, np.reciprocal(dist))
    else:
        # Avoid potentially-faulty reward function.
        if dist >= (min_dist - 0.0001):
            return (min_dist, -.1)
        else:
            return (dist, 2 + np.reciprocal(dist))

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

        self.color = PURPLE
        self.collide = True
        self.silent = silent
        self.size = 0.075
        self.accel = 5.0
        self.max_speed = 10.0
        self.name = "hmni %d" % self.identity

        self.unrealized_reward = 0

    def reward(self, world, scenario):
        # All agents stay in-bounds
        def bound1(x):
            if x < .9:
                return 0
            elif x < 1.0:
                return 100*(x - .9)
            else:
                return min(exp(2*x - 2), 200)

        reward = self.unrealized_reward
        self.unrealized_reward = 0
        for p in range(world.dim_p):
            x = abs(self.state.p_pos[p])
            reward -= bound1(x)
        return reward

    def reset(self):
        self.unrealized_reward = 0
        self.carrying_food = self.stole_food = False

class Pleb(HelloMyNameIs):
    def __init__(self, food_source, nest):
        super(Pleb, self).__init__()
        self.true_color = PURPLE
        self.name = "pleb %d" % self.identity
        self.closest_approach = FAR_AWAY

        self.food_source = food_source
        self.nest = nest

        self.objectives = [food_source, nest]
        self.objective_rewards = [self._pickup_food, self._dropoff_food]
        self.current_objective = 0

    def reward(self, world, scenario):
        reward = super(Pleb, self).reward(world, scenario)

        # If the agent is fortunate enough to live in a world where it
        # is touching both a food source and a nest at the same time,
        # it should not be rewarded since (presumably) nothing can be
        # learned.  We simply wait for the simulation to expire.
        #-if is_collision(self, self.nest) and is_collision(self, self.food_source):
        #-    return 0

        obj = self.current_objective
        spatial_objective = self.objectives[obj]

        # Get a reward if we are closer to our spatial objective than last time.
        self.closest_approach, r = \
            reward_closer(self, spatial_objective, self.closest_approach)
        reward += r

        # Get a reward if we have achieved our current objective.
        if is_collision(self, spatial_objective):
            # Share the rewards with all other plebs.  This will allow other
            # plebs to take on roles that may not directly involve carrying food
            # (i.e. guard, warrior, or decoy).
            r = self.objective_rewards[obj]()
            for pleb in scenario.plebs:
                pleb.unrealized_reward += r
            reward += r
            spatial_objective = self._next_objective()

        # Get a reward if we are closer to our spatial objective than last time.
        # Note: this may be a new spatial objective.
        self.closest_approach, r = \
            reward_closer(self, spatial_objective, self.closest_approach)
        reward += r
        return reward

    def reset(self):
        super(Pleb, self).reset()
        self.current_objective = 0

    def _dropoff_food(self):
        self.carrying_food    = False
        self.closest_approach = FAR_AWAY
        return REWARD_DROPOFF_FOOD

    def _pickup_food(self):
        self.carrying_food    = True
        self.closest_approach = FAR_AWAY
        return REWARD_PICKUP_FOOD

    def _next_objective(self):
        obj = self.current_objective
        self.current_objective = obj = (1 + obj) % len(self.objectives)
        return self.objectives[obj]

    def _lose_food(self):
        self.unrealized_reward -= REWARD_STEAL_FOOD
        self.carrying_food      = False
        self.closest_approach   = FAR_AWAY
        self._next_objective()

class Spy(HelloMyNameIs):
    def __init__(self):
        super(Spy, self).__init__()
        self.true_color = CYAN
        self.name       = "spy %d" % self.identity

    def reset(self):
        super(Spy, self).reset()

    def reward(self, world, scenario):
        reward = super(Spy, self).reward(world, scenario)
        pleb_proximity_reward = 0
        pleb_theft_reward = 0
        for pleb in scenario.plebs:
            if is_collision(self, pleb):
                if pleb.carrying_food:
                    pleb._lose_food()
                    pleb_theft_reward += REWARD_STEAL_FOOD
            else:
                _, r = reward_closer(self, pleb, FAR_AWAY)
                pleb_proximity_reward += r

        # The spy is rewarded for all of the food it is able to steal.
        reward += pleb_theft_reward

        # All plebs share the pain of losing food to a Spy.  This will encourage
        # other plebs to take on roles that may not directly involve carrying
        # food (i.e. guard, warrior, or decoy).
        for pleb in scenario.plebs:
            pleb.unrealized_reward -= pleb_theft_reward

        # Bound the proximity rewards.
        reward += min(7, pleb_proximity_reward)
        return reward

class Scenario(BaseScenario):
    def make_world(self, use_reputation=True):
        self.world = world = World()
        self.use_reputation = use_reputation

        world.dim_c = 2

        self.food_source = food_source = Landmark()
        food_source.name = 'food-source'
        food_source.collide = False
        food_source.movable = False
        food_source.size = .09
        food_source.boundary = False
        food_source.color = GREEN

        self.nest = nest = Landmark()
        nest.name = 'nest'
        nest.collide = False
        nest.movable = False
        nest.size = .09
        nest.boundary = False
        nest.color = BROWN

        world.landmarks = [food_source, nest]

        # Add agents.
        self.plebs = [Pleb(food_source, nest) for i in range(2)]
        self.spies = [Spy() for i in range(1)]
        world.agents = self.plebs + self.spies

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # Set random initial states for all agents.
        for pleb in self.plebs:
            pleb.reset()
            spawn_entity(pleb, world)

        for spy in self.spies:
            spy.reset()
            spawn_entity(spy, world)

        # Set random positions for landmarks.
        spawn_entity(self.nest, world)
        spawn_entity(self.food_source, world)

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

        if agent.carrying_food:
            food_status = np.array([1., 0., 0., 0.])
        else:
            food_status = np.array([0., 0., 0., 0.])

        return np.concatenate([agent.state.p_vel] + \
                              [agent.state.p_pos] + \
                              [food_status] + \
                              landmark_pos + other_pos + other_vel)

    def reward(self, agent, world):
        return agent.reward(world, self)

    #-def benchmark_data(self, agent, world):
    #-    pass
