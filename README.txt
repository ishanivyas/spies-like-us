-*- tab-width:4 -*-
Directory layout:
	README.txt	-- This file.
	maddpg/spies-like-us/experiments/	-- The saved results of experiments.
		1.0	-- The first experiment.
		1.1	-- The second experiment.
		...
		1.x/
			checkpoint
			.meta
			.index
			.data-00000-of-00001

Training new experiments:
	$ python3 train.py --scenario=spies-like-us            ;# Train a new policy.
	$ python3 train.py --scenario=spies-like-us --restore  ;# Do more training on an existing policy.

Observing the results of the trained experiments:
	$ python3 train.py --scenario=spies-like-us --restore --display


spies-like-us-1.0:
	Simple Pleb reward:
		* Fixed amount of points for picking up food.
		* Higher fixed amount of points for dropping off food.
		* Points in inverse proportion to distance from current objective (either food source or nest).

spies-like-us-1.1:
	Noticed lots of hesitation as Plebs got near food source and nest.  Tried to compensate.

spies-like-us-1.2:
	Added noose so that Plebs must get closer to objective in order to receive any more rewards.  This prohibits procrastination/cheating.

spies-like-us-1.3:
	Adding Spies to the mix:
		* If they contact a Pleb, they steal its food and gain a reward.
		* Rewarded in inverse proportion to distance to all Plebs (up to a bound).
	Outcome:
		* The spy ambushes the Plebs near the food source.
		* The Plebs do not develop any compensating behavior.
	Interpretation:
		The Plebs are happy enough to score rewards by only picking up food
		because the Spy's reward for taking the food does not undo all of the
		Plebs' reward for picking up food, this is a beneficial symbiotic
		relationship.

spies-like-us-1.4:
	* Penalize all Plebs when a Spy steals from a Pleb.
	* Reward all Plebs when any Pleb picks up food or drops it off.

* Watchpoint: Plebs may rely upon the position of an agent in the observation vector in order to assign reputation, i.e. they learn that certain slots holding observations of agents will always hold spies.  They do not learn to adapt when a spy shows up in a different slot.

Observations:
	* Pathological reward function that was found.
	* Pathological reward function seems to imply that Plebs learn to count steps, so if they are too far away from their objective, they know they can gain more points by procrastinating near the objectie than by picking-up/dropping-off and accelerating toward the next objective.
